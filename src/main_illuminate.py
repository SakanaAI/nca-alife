import argparse
import copy
import os
from collections import defaultdict
from functools import partial

import evosax
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split
from PIL import Image
from tqdm.auto import tqdm

import util
from clip_jax import MyFlaxCLIP, MyFlaxDinov2, MyFlaxPixels
from create_sim import create_sim, rollout_and_embed_simulation, FlattenSimulationParameters

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")
group.add_argument("--sim", type=str, default='boids')

group = parser.add_argument_group("data")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group = parser.add_argument_group("optimization")
group.add_argument("--k_nbrs", type=int, default=2)
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
    sim = create_sim(args.sim)
    sim = FlattenSimulationParameters(sim)

    if args.clip_model =='clip-vit-base-patch32':
        clip_model = MyFlaxCLIP(args.clip_model)
    elif args.clip_model == 'dinov2-base':
        clip_model = MyFlaxDinov2(args.clip_model, features='pooler')
    elif args.clip_model == 'pixels':
        clip_model = MyFlaxPixels()
    else:
        raise ValueError(f"clip_model {args.clip_model} not recognized")

    rollout_fn_ = partial(rollout_and_embed_simulation, sim=sim, clip_model=clip_model, rollout_steps=sim.sim.rollout_steps, n_rollout_imgs='final')
    rng = jax.random.PRNGKey(args.seed)

    rollout_fn = jax.jit(lambda rng, p: dict(params=p, **rollout_fn_(rng, p)))

    rng, _rng = split(rng)
    params_init = 0.*jax.random.normal(_rng, (args.pop_size, sim.n_params))
    pop = [rollout_fn(_rng, p) for p in tqdm(params_init)]
    pop = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *pop)

    @jax.jit
    def do_iter(pop, rng):
        rng, _rng = split(rng)
        idx_p1, idx_p2 = jax.random.randint(_rng, (2, args.bs), minval=0, maxval=args.pop_size)
        params_parent1, params_parent2 = pop['params'][idx_p1], pop['params'][idx_p2]  # bs D
        rng, _rng1, _rng2 = split(rng, 3)
        noise1, noise2 = jax.random.normal(_rng1, (args.bs, sim.n_params)), jax.random.normal(_rng2, (args.bs, sim.n_params))
        params_children = params_parent1 + args.sigma1*noise1 + args.sigma2*(params_parent2-params_parent1)*noise2

        rng, _rng = split(rng)
        children = jax.vmap(rollout_fn)(split(_rng, args.bs), params_children)

        pop = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *[pop, children])

        X = pop['z'] # (pop_size+bs) D
        print(X.shape)
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

            print(jax.tree_map(lambda x: x.shape, pop))
            util.save_pkl(args.save_dir, "pop", jax.tree.map(lambda x: np.array(x), pop))
            
            # plt.figure(figsize=(10, 5))
            # plt.subplot(211)
            # plt.plot(data_save['loss'], color='green', label='loss')
            # plt.axhline(data_save['loss'][0], color='r', linestyle='dashed', label='initial loss')
            # plt.legend()
            # img = pop_save['img_final'] # pop_size H W C
            # img = jnp.pad(img, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=.5)
            # plt.subplot(212); plt.imshow(rearrange(img[::(img.shape[0]//16), :, :, :], "(R C) H W D -> (R H) (C W) D", R=2))
            # plt.savefig(f'{args.save_dir}/overview_{i_iter:06d}.png')
            # plt.close()

if __name__ == '__main__':
    main(parse_args())
