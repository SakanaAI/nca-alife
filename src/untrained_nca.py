import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

import flax.linen as nn

import models
import util

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
# group.add_argument("--load_ckpt", type=str, default=None)
group.add_argument("--save_dir", type=str, default=None)
# group.add_argument("--save_ckpt", type=lambda x: x=="True", default=False)

group = parser.add_argument_group("data")
group.add_argument("--height", type=int, default=256)
group.add_argument("--width", type=int, default=256)
group.add_argument("--dt", type=float, default=0.01)
group.add_argument("--p_drop", type=float, default=0.0)
group.add_argument("--init_state", type=str, default="randn")

group = parser.add_argument_group("model")
group.add_argument("--n_layers", type=int, default=2)
group.add_argument("--d_state", type=int, default=16)
group.add_argument("--d_embd", type=int, default=32)
group.add_argument("--kernel_size", type=int, default=3)
group.add_argument("--nonlin", type=str, default="gelu")

group = parser.add_argument_group("rollout")
group.add_argument("--n_steps", type=int, default=int(10e9))
group.add_argument("--n_steps_chunk", type=int, default=int(100e3))
group.add_argument("--bs", type=int, default=1)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


class NCA(nn.Module):
    n_layers: int
    d_state: int
    d_embd: int
    kernel_size: int = 3
    nonlin: str = 'gelu'
    
    @nn.compact
    def __call__(self, x, train=False):  
        for _ in range(self.n_layers):
            x = nn.Conv(features=self.d_embd, kernel_size=(self.kernel_size, self.kernel_size), feature_group_count=1,
                        kernel_init=nn.initializers.normal(1.), bias_init=nn.initializers.normal(1.))(x)
            # x = nn.LayerNorm()(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)  
            x = getattr(nn, self.nonlin)(x)
        x = nn.Conv(features=self.d_state, kernel_size=(self.kernel_size, self.kernel_size), feature_group_count=1,
                    kernel_init=nn.initializers.normal(1.), bias_init=nn.initializers.normal(1.))(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)  
        return x


def main(args):
    util.save_json(args.save_dir, "args", vars(args))

    rng = jax.random.PRNGKey(args.seed)
    nca = NCA(n_layers=args.n_layers, d_state=args.d_state, d_embd=args.d_embd, kernel_size=args.kernel_size,
                     nonlin=args.nonlin)
    init_state_fn = partial(models.sample_init_state, height=args.height, width=args.width, d_state=args.d_state,
                            init_state=args.init_state)
    rollout_fn = partial(models.nca_rollout, nca, rollout_steps=args.n_steps_chunk, dt=args.dt, p_drop=args.p_drop,
                         vid_type='none')

    dummy_state = init_state_fn(rng)
    rng, _rng = split(rng)
    params = nca.init(_rng, dummy_state)
    print(jax.tree.map(lambda x: x.shape, params))
    print("Number of parameters:", sum([p.size for p in jax.tree.flatten(params)[0]]))

    def init_batch_stats(params, _rng):  
        x = jax.random.normal(_rng, (16, args.height, args.width, args.d_state))
        y, updates = nca.apply(params, x, train=True, mutable=['batch_stats'])  
        params['batch_stats'] = updates['batch_stats']  
        # y = net.apply(params, x)  
        return params

    def check_distributions(_rng):
        x = jax.random.normal(_rng, (16, args.height, args.width, args.d_state))
        y, state = jax.vmap(partial(nca.apply, capture_intermediates=True, mutable=["intermediates"]), in_axes=(None, 0))(params, x)
        print(jax.tree.map(lambda x: f"{x.mean().item(): 0.4f}, {x.std().item():0.4f}", state['intermediates']))

    print('-'*50)
    rng, _rng = split(rng)
    check_distributions(_rng)
    print('-'*50)

    rng, _rng = split(rng)
    params = init_batch_stats(params, _rng)
    
    print('-'*50)
    rng, _rng = split(rng)
    check_distributions(_rng)
    print('-'*50)
    print(jax.tree.map(lambda x: x.shape, params))

    p_drop, dt, H, W = args.p_drop, args.dt, args.height, args.width

    def forward_step(state, _rng):
        H, W, D = state.shape
        dstate = nca.apply(params, state)
        drop_mask = jax.random.uniform(_rng, (H, W, 1)) < p_drop
        next_state = state + dt * dstate * (1. - drop_mask)
        next_state = next_state / jnp.linalg.norm(next_state, axis=-1, keepdims=True)
        return next_state, None

    def forward_chunk(state, _rng):
        return jax.lax.scan(forward_step, state, split(_rng, args.n_steps_chunk))

    batch_forward_chunk = jax.jit(jax.vmap(forward_chunk))

    def save_ckpt_data(_rng, i_iter, state):
        if args.save_dir is None:
            return

        util.save_pkl(args.save_dir, 'params', jax.tree.map(lambda x: np.array(x), params))
        util.save_pkl(args.save_dir, f'state_{i_iter:011d}', jax.tree.map(lambda x: np.array(x), state))
        util.save_pkl(args.save_dir, f'state_latest', jax.tree.map(lambda x: np.array(x), state))

    pbar = tqdm(range(args.n_steps))

    rng, _rng = split(rng)
    state = jax.vmap(init_state_fn)(split(_rng, args.bs))

    params = jax.tree.map(lambda x: x.astype(jnp.float16), params)
    state = jax.tree.map(lambda x: x.astype(jnp.float16), state)

    
    for i_iter in range(0, args.n_steps, args.n_steps_chunk):
        rng, _rng = split(rng)
        state, vid = batch_forward_chunk(state, split(_rng, args.bs))

        pbar.update(args.n_steps_chunk)

        if i_iter % (args.n_steps // 100) == 0:
            save_ckpt_data(_rng, i_iter, state)
    save_ckpt_data(_rng, args.n_steps, state)


if __name__ == "__main__":
    main(parse_args())
