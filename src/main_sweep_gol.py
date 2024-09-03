import argparse
import copy
import os
from collections import defaultdict
from functools import partial

import evosax
import imageio
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split
from PIL import Image
from tqdm.auto import tqdm

import util
from clip_jax import MyFlaxCLIP
from models.models_gol import GameOfLife

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")
group.add_argument("--grid_size", type=int, default=64)
group.add_argument("--rollout_steps", type=int, default=4096)

group = parser.add_argument_group("data")
group.add_argument("--n_rollout_imgs", type=int, default=32)
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=512)
group.add_argument("--start", type=int, default=0) # start range for params search
group.add_argument("--end", type=int, default=262144) # end range for params search


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    sim = GameOfLife(grid_size=args.grid_size)
    clip_model = MyFlaxCLIP(args.clip_model)

    rng = jax.random.PRNGKey(args.seed)

    def calc_loss(rng, params):
        def step(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, state
        state_init = sim.init_state(rng, params)
        state_final, state_vid = jax.lax.scan(step, state_init, split(rng, args.rollout_steps))

        sr = args.rollout_steps//args.n_rollout_imgs
        idx_downsample = jnp.arange(0, args.rollout_steps, sr)
        state_vid = jax.tree.map(lambda x: x[idx_downsample], state_vid) # downsample
        print('state_vid', state_vid.shape)
        vid = jax.vmap(partial(sim.render_state, params=params, img_size=224))(state_vid) # T H W C
        print('vid', vid.shape)
        z_img = jax.vmap(clip_model.embed_img)(vid) # T D
        print('z_img', z_img.shape)

        # --------- CLIP Novelty ---------
        scores_novelty = (z_img @ z_img.T) # T T
        scores_novelty = jnp.tril(scores_novelty, k=-1)
        loss_novelty = scores_novelty.max(axis=-1) # T
        print('loss_novelty', loss_novelty.shape)

        # --------- Manual Novelty ---------
        scores_novelty = jnp.abs(state_vid[None, :] - state_vid[:, None]).mean(axis=(-1, -2)) # T T
        scores_novelty = jnp.tril(scores_novelty, k=-1)
        loss_novelty_manual = scores_novelty.max(axis=-1) # T
        print('loss_novelty_manual', loss_novelty_manual.shape)
        return dict(loss_novelty=loss_novelty, loss_novelty_manual=loss_novelty_manual, z_img_final=z_img[-1])

    @jax.jit
    def do_iter(params, rng):
        calc_loss_v = jax.vmap(calc_loss, in_axes=(0, None))
        data = calc_loss_v(split(rng, args.bs), params)
        print('data', jax.tree.map(lambda x: x.shape, data))

        data = dict(loss_novelty=data['loss_novelty'].mean(axis=0),
                    loss_novelty_manual=data['loss_novelty_manual'].mean(axis=0),
                    z_img_final=data['z_img_final'][args.bs//2])
        print('data', jax.tree.map(lambda x: x.shape, data))
        return data

    args.n_iters = args.end - args.start
    all_params = np.arange(args.start, args.end)
    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        di = do_iter(all_params[i_iter], _rng)
        data.append(di)

        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            util.save_pkl(args.save_dir, "all_params", all_params)

if __name__ == '__main__':
    main(parse_args())
