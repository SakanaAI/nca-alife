import argparse
from functools import partial

import jax
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm

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
group.add_argument("--height", type=int, default=32)
group.add_argument("--width", type=int, default=32)
group.add_argument("--dt", type=float, default=0.01)
group.add_argument("--p_drop", type=float, default=0.0)
group.add_argument("--init_state", type=str, default="point")
group.add_argument("--rollout_steps", type=int, default=64)
group.add_argument("--target_img_path", type=str, default=None)

group = parser.add_argument_group("model")
group.add_argument("--n_layers", type=int, default=1)
group.add_argument("--d_state", type=int, default=16)
group.add_argument("--d_embd", type=int, default=64)
group.add_argument("--kernel_size", type=int, default=3)
group.add_argument("--nonlin", type=str, default="gelu")

group = parser.add_argument_group("rollout")
group.add_argument("--n_steps", type=int, default=100000000)
group.add_argument("--n_steps_chunk", type=int, default=1000)
group.add_argument("--bs", type=int, default=1)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


def main(args):
    util.save_json(args.save_dir, "args", vars(args))

    rng = jax.random.PRNGKey(args.seed)
    nca = models.NCA(n_layers=args.n_layers, d_state=args.d_state, d_embd=args.d_embd, kernel_size=args.kernel_size,
                     nonlin=args.nonlin)
    init_state_fn = partial(models.sample_init_state, height=args.height, width=args.width, d_state=args.d_state,
                            init_state=args.init_state)
    rollout_fn = partial(models.nca_rollout, nca, rollout_steps=args.rollout_steps, dt=args.dt, p_drop=args.p_drop,
                         vid_type='none')

    dummy_state = init_state_fn(rng)
    rng, _rng = split(rng)
    params = nca.init(_rng, dummy_state)
    print(jax.tree.map(lambda x: x.shape, params))
    print("Number of parameters:", sum([p.size for p in jax.tree.flatten(params)[0]]))

    def save_ckpt_data(_rng, i_iter):
        if args.save_dir is None:
            return

    pbar = tqdm(range(args.n_steps))

    rng, _rng = split(rng)
    state = jax.vmap(init_state_fn)(split(_rng, args.bs))
    for i_iter in range(0, args.n_iters, args.n_iters_chunk):
        rng, _rng = split(rng)
        state, vid = jax.vmap(rollout_fn, in_axes=(None, 0, 0))(params, split(_rng, args.bs), state)

        pbar.update(args.n_iters_chunk)

        if i_iter % (args.n_iters // 5) == 0:
            save_ckpt_data(_rng, i_iter)
    save_ckpt_data(_rng, args.n_iters)


if __name__ == "__main__":
    main(parse_args())
