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
from models_nca import NCA

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")
group.add_argument("--grid_size", type=int, default=128)
group.add_argument("--d_state", type=int, default=1)
group.add_argument("--rollout_steps", type=int, default=1024)

group = parser.add_argument_group("data")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group = parser.add_argument_group("optimization")
group.add_argument("--k_nbrs", type=int, default=1)
group.add_argument("--bs", type=int, default=32)
group.add_argument("--pop_size", type=int, default=1024)
group.add_argument("--n_iters", type=int, default=10000)
group.add_argument("--sigma1", type=float, default=0.1)
group.add_argument("--sigma2", type=float, default=0.)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    sim = NCA(grid_size=args.grid_size, d_state=args.d_state, p_drop=0.0)
    clip_model = MyFlaxCLIP(args.clip_model)

    rng = jax.random.PRNGKey(args.seed)
    param_reshaper = evosax.ParameterReshaper(sim.default_params(rng))

    @jax.jit
    def unroll_params(rng, params):
        def step(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, state
        state_init = sim.init_state(rng, params)
        state_final, state_vid = jax.lax.scan(step, state_init, split(rng, args.rollout_steps))

        img_init = sim.render_state(state_init, params=params, img_size=224)
        img_final = sim.render_state(state_final, params=params, img_size=224)
        z_img_final = clip_model.embed_img(img_final) # D
        return dict(img_init=img_init, img_final=img_final, z_img_final=z_img_final)

    rng, _rng = split(rng)
    pop = []
    for i in tqdm(range(args.pop_size)):
        params = jnp.zeros(param_reshaper.total_params)
        rng, _rng = split(rng)
        unroll_data = unroll_params(_rng, param_reshaper.reshape_single(params))
        pop.append({"params": params, **unroll_data})
    pop = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *pop)

    @jax.jit
    def do_iter(pop, rng):
        rng, _rng = split(rng)
        idx_p1, idx_p2 = jax.random.randint(_rng, (2, args.bs), minval=0, maxval=args.pop_size)
        parent1, parent2 = pop['params'][idx_p1], pop['params'][idx_p2]  # bs D
        rng, _rng1, _rng2 = split(rng, 3)
        noise1, noise2 = jax.random.normal(_rng1, (args.bs, param_reshaper.total_params)), jax.random.normal(_rng2, (args.bs, 1))
        children = parent1 + args.sigma1*noise1 + args.sigma2*(parent2-parent1)*noise2

        rng, _rng = split(rng)
        unroll_data = jax.vmap(unroll_params)(split(_rng, args.bs), param_reshaper.reshape(children))
        children = {"params": children, **unroll_data}

        pop = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *[pop, children])

        X = pop['z_img_final'] # (pop_size+bs) D
        D = -X@X.T # (pop_size+bs) (pop_size+bs)
        D = D.at[jnp.arange(args.pop_size+args.bs), jnp.arange(args.pop_size+args.bs)].set(jnp.inf)

        to_kill = jnp.zeros(args.bs, dtype=int) # indices of pop to kill

        def kill_least(carry, _):
            D, to_kill, i = carry

            tki = D.sort(axis=-1)[:, :args.k_nbrs].mean(axis=-1).argmin()
            D = D.at[:, tki].set(jnp.inf)
            D = D.at[tki, :].set(jnp.inf)
            to_kill = to_kill.at[i].set(tki)

            return (D, to_kill, i+1), None

        (D, to_kill, _), _ = jax.lax.scan(kill_least, (D, to_kill, 0), None, length=args.bs)
        to_keep = jnp.setdiff1d(jnp.arange(args.pop_size+args.bs), to_kill, assume_unique=True, size=args.pop_size)

        pop = jax.tree.map(lambda x: x[to_keep], pop)
        D = D[to_keep, :][:, to_keep]
        loss = -D.min(axis=-1).mean()
        return pop, dict(loss=loss)

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        pop, di = do_iter(pop, rng)

        data.append(di)
        pbar.set_postfix(loss=di['loss'].item())
        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)

            pop_save = jax.tree.map(lambda x: x, pop)
            pop_save['img_init'] = (pop_save['img_init']*255).astype(jnp.uint8)
            pop_save['img_final'] = (pop_save['img_final']*255).astype(jnp.uint8)
            pop_save = jax.tree.map(lambda x: np.array(x), pop_save)
            util.save_pkl(args.save_dir, "pop", pop_save)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(211)
            plt.plot(data_save['loss'], color='green', label='loss')
            plt.axhline(data_save['loss'][0], color='r', linestyle='dashed', label='initial loss')
            plt.legend()
            img = pop_save['img_final'] # pop_size H W C
            img = jnp.pad(img, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=.5)
            plt.subplot(212); plt.imshow(rearrange(img[::(img.shape[0]//16), :, :, :], "(R C) H W D -> (R H) (C W) D", R=2))
            plt.savefig(f'{args.save_dir}/overview_{i_iter:06d}.png')
            plt.close()

if __name__ == '__main__':
    main(parse_args())
