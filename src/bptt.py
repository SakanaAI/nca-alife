import argparse
import os
import pickle

import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm
from einops import rearrange
from functools import partial

import models
import util

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
# group.add_argument("--load_ckpt", type=str, default=None)
group.add_argument("--save_dir", type=str, default=None)
# group.add_argument("--save_ckpt", type=lambda x: x=="True", default=False)
group.add_argument("--n_iters_chunk", type=int, default=1)

group = parser.add_argument_group("data")
group.add_argument("--height", type=int, default=32)
group.add_argument("--width", type=int, default=32)
group.add_argument("--dt", type=float, default=0.01)
group.add_argument("--p_drop", type=float, default=0.0)
group.add_argument("--init_state", type=str, default="point")
group.add_argument("--rollout_steps", type=int, default=64)
group.add_argument("--target_img_path", type=str, default=None)
group.add_argument("--apply_loss", type=str, default="all")  # all or last

group = parser.add_argument_group("model")
group.add_argument("--n_layers", type=int, default=1)
group.add_argument("--d_state", type=int, default=16)
group.add_argument("--d_embd", type=int, default=64)
group.add_argument("--kernel_size", type=int, default=3)
group.add_argument("--nonlin", type=str, default="gelu")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=10000)
group.add_argument("--bs", type=int, default=1)
group.add_argument("--lr", type=float, default=3e-4)
group.add_argument("--lr_schedule", type=str, default="constant")  # constant or cosine_decay
group.add_argument("--weight_decay", type=float, default=0.)
group.add_argument("--clip_grad_norm", type=float, default=1.)

# group = parser.add_argument_group("rollout")
# group.add_argument("--env_id", type=str, default=None)
# group.add_argument("--n_envs", type=int, default=64)
# group.add_argument("--n_iters_rollout", type=int, default=1000)
# group.add_argument("--video", type=lambda x: x=='True', default=False)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


def load_img(img_path, height=32, width=32):
    from PIL import Image
    img = Image.open(img_path).convert('RGB').resize((height, width))
    img = jnp.array(img, dtype=np.float32)/255.
    return img
    

def main(args):
    util.save_json(args.save_dir, "args", vars(args))

    rng = jax.random.PRNGKey(args.seed)
    nca = models.NCA(n_layers=args.n_layers, d_state=args.d_state, d_embd=args.d_embd, kernel_size=args.kernel_size, nonlin=args.nonlin)
    init_state_fn = partial(models.sample_init_state, height=args.height, width=args.width, d_state=args.d_state, init_state=args.init_state)
    rollout_fn = partial(models.nca_rollout, nca, rollout_steps=args.rollout_steps, dt=args.dt, p_drop=args.p_drop)

    dummy_state = init_state_fn(rng)
    rng, _rng = split(rng)
    params = nca.init(_rng, dummy_state)
    print(jax.tree.map(lambda x: x.shape, params))
    print("Number of parameters:", sum([p.size for p in jax.tree.flatten(params)[0]]))
    
    target_img = load_img(args.target_img_path, height=args.height, width=args.width)
    assert target_img.shape==(args.height, args.width, 3)

    if args.lr_schedule == "constant":
        lr_schedule = optax.constant_schedule(args.lr)
    elif args.lr_schedule == "cosine_decay":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0., peak_value=args.lr,
            warmup_steps=args.n_iters//100, decay_steps=args.n_iters,
            end_value=args.lr/10, exponent=1.0
        )
    tx = optax.chain(
        optax.clip_by_global_norm(args.clip_grad_norm),
        optax.adamw(lr_schedule, weight_decay=args.weight_decay, eps=1e-8)
    )
    train_state = TrainState.create(apply_fn=nca.apply, params=params, tx=tx)

    def loss_fn(nca_params, batch):
        _rng, state = batch['_rng'], batch['state']
        next_state, vid = jax.vmap(rollout_fn, in_axes=(None, 0, 0))(nca_params, _rng, state)
        if args.apply_loss=='all':
            mse = ((vid[:, :, ...]-target_img)**2)
        elif args.apply_loss=='last':
            mse = ((vid[:, -1, ...]-target_img)**2)
        loss = mse.mean()
        return loss, dict(loss=loss, mse=mse, vid=vid)

    @jax.jit
    def train_step(train_state, _rng):
        _rng1, _rng2 = split(_rng)
        batch = dict(_rng=split(_rng1, args.bs), state=jax.vmap(init_state_fn)(split(_rng2, args.bs)))
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def save_ckpt_data(_rng, i_iter, train_state):
        if args.save_dir is None:
            return

        util.save_pkl(args.save_dir, "losses", losses)

        state = init_state_fn(_rng)
        next_state, vid = rollout_fn(train_state.params, _rng, state)
        i_step = jnp.linspace(0, args.rollout_steps, 5).astype(int).clip(0, args.rollout_steps-1)
        vid = np.array(vid[i_step].clip(0, 1))

        import matplotlib.pyplot as plt
        plt.imshow(rearrange(vid, "(R C) H W D -> (R H) (C W) D", R=1))
        plt.title(f"Steps: {i_step}")
        plt.savefig(f"{args.save_dir}/rollout_{i_iter:07d}.png")
        plt.close()

        plt.subplot(131)
        plt.imshow(vid[-1])
        plt.subplot(132)
        plt.imshow(target_img)
        plt.subplot(133)
        plt.imshow(((vid[-1]-target_img)**2).mean(axis=-1))
        plt.colorbar()
        plt.savefig(f"{args.save_dir}/error.png")
        plt.close()

    losses = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in range(0, args.n_iters, args.n_iters_chunk):
        rng, _rng = split(rng)
        train_state, losses_i = jax.lax.scan(train_step, train_state, split(_rng, args.n_iters_chunk))
        losses.extend(losses_i.tolist())
        pbar.set_postfix(log10_loss=np.log10(losses_i.mean()).item())
        pbar.update(args.n_iters_chunk)


        if i_iter % (args.n_iters // 5) == 0:
            save_ckpt_data(_rng, i_iter, train_state)
    save_ckpt_data(_rng, args.n_iters, train_state)


if __name__ == "__main__":
    main(parse_args())
