import numpy as np

from tqdm.auto import tqdm
import copy
from einops import rearrange, reduce, repeat

import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat

from functools import partial

from PIL import Image
from transformers import AutoProcessor, CLIPModel

import matplotlib.pyplot as plt
import util
import argparse
from collections import defaultdict

import imageio

import numpy as np
from tqdm.auto import tqdm

from models_particle_life import ParticleLife
from clip_jax import MyFlaxCLIP

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoProcessor, FlaxCLIPModel

# from evosax import CMA_ES, SimpleGA

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--dtype", type=str, default='float32')
group.add_argument("--device", type=str, default='cuda:0')
group.add_argument("--n_iters_scan", type=int, default=1)

group = parser.add_argument_group("model")
group.add_argument("--n_particles", type=int, default=1000)
group.add_argument("--n_colors", type=int, default=3)

group.add_argument("--n_dims", type=int, default=2)
group.add_argument("--dt", type=float, default=0.001)  # 0.01 sometimes too much
group.add_argument("--half_life", type=float, default=0.04)
group.add_argument("--rmax", type=float, default=0.1)

group = parser.add_argument_group("data")
group.add_argument("--prompt", type=str, default="a diverse ecosystem of cells moving around")
group.add_argument("--augs", type=str, default='crop+pers')
group.add_argument("--aug_crop_scale", type=float, default=1.)

group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group.add_argument("--coef_alignment", type=float, default=1.)
# group.add_argument("--coef_novelty", type=float, default=0.)

group = parser.add_argument_group("optimization")
group.add_argument("--rollout_steps", type=int, default=3000)

group.add_argument("--bs", type=int, default=8)
group.add_argument("--lr", type=float, default=3e-4)
group.add_argument("--n_iters", type=int, default=10000)
group.add_argument("--clip_grad_norm", type=float, default=1.)


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

# to generate beautiful images:
# render_fn_clip = partial(plife.render_state_heavy, img_size=1024, radius=2., sharpness=10.)
# particles: 10000, rollout_len 1000

def main(args):
    rng = jax.random.PRNGKey(args.seed)

    plife = ParticleLife(args.n_particles, args.n_colors, n_dims=args.n_dims)
    clip_model = MyFlaxCLIP(args.clip_model)
    z_text = clip_model.embed_text([args.prompt])

    # render_fn_clip = partial(plife.render_state_heavy, img_size=1024, radius=2., sharpness=10.)
    # render_fn_vid = partial(plife.render_state_heavy, img_size=256, radius=0.5, sharpness=10.)
    # render_fn_clip = partial(plife.render_state_heavy, img_size=224, radius=0.4375, sharpness=3.)
    # render_fn_clip = partial(plife.render_state_light, img_size=224, radius=0)
    render_fn_clip = partial(plife.render_state_heavy, img_size=224, radius=8e-3, sharpness=5e3)

    render_fn_clip = jax.jit(render_fn_clip)

    def rollout_plife(rng, env_params, rollout_steps=args.rollout_steps, return_statevid=False):
        state = plife.get_random_init_state(rng)
        def step(state, _):
            next_state = plife.forward_step(state, env_params)
            return next_state, (state if return_statevid else None)
        state, statevid = jax.lax.scan(step, state, length=rollout_steps)
        return state, statevid

    def mutate(rng, x):
        x = {k: v for k, v in x.items()}
        alpha = x['alpha']
        rng, _rng = split(rng)
        mask = (jax.random.uniform(_rng, alpha.shape) < 0.1).astype(jnp.float32)
        noise = jax.random.uniform(rng, alpha.shape, minval=-1., maxval=1.)
        x['alpha'] = (1.-mask)*alpha + mask*noise
        return x
        
    def calc_fitness(rng, x):
        env_params = x
        state, _ = rollout_plife(rng, env_params, return_statevid=False)
        img = render_fn_clip(state, env_params)
        z_img = clip_model.embed_img(img)
        return (z_text @ z_img.T).mean(), img

    def gen_random_individual(rng):
        return plife.get_default_env_params()

    rng, _rng = split(rng)
    population = jax.vmap(gen_random_individual)(split(_rng, args.bs))

    def do_iter(population, rng):
        rng, _rng = split(rng)
        fitness, img = jax.vmap(calc_fitness)(split(_rng, args.bs), population)
        idx = jnp.argsort(fitness, descending=True)
        elite = jax.tree.map(lambda x: x[idx[0]], population)

        rng, _rng = split(rng)
        parent_idx = idx[jax.random.randint(_rng, (args.bs,), minval=0, maxval=args.bs//2)]
        next_population = jax.tree.map(lambda x: x[parent_idx], population)

        rng, _rng = split(rng)
        next_population = jax.vmap(mutate)(split(_rng, args.bs), next_population)

        next_population = jax.tree.map(lambda x, y: x.at[0].set(y), next_population, elite)
        return next_population, dict(population=population, fitness=fitness, img=img)
    do_iter = jax.jit(do_iter)

    # def save_ckpt(save_dir, i_iter, population, metrics):
        # util.save_ckpt(save_dir, i_iter, dict(population=population, metrics=metrics))

    fitnesses = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        population, metrics = do_iter(population, _rng)
        # population, metrics = jax.lax.scan(do_iter, population, split(_rng, 1))

        fitnesses.append(metrics['fitness'])
        pbar.set_postfix(mean_fitness=metrics['fitness'].mean(), max_fitness=metrics['fitness'].max())

        if args.save_dir is not None and (i_iter%(args.n_iters//10)==0 or i==args.n_iters-1):
            util.save_pkl(args.save_dir, 'fitnesses', jax.tree.map(lambda x: np.array(x), fitnesses))
            util.save_pkl(args.save_dir, 'population', jax.tree.map(lambda x: np.array(x), population))

            imgs = metrics['img'][:8]
            plt.figure(figsize=(40, 20))
            for i, img in enumerate(imgs):
                fitness = metrics['fitness'][i]
                plt.subplot(2, 4, i+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"{fitness:.4f}")
            plt.suptitle(f"Prompt: {args.prompt},  Iteration: {i_iter:06d}")
            plt.tight_layout()
            plt.savefig(f"{args.save_dir}/imgs_{i_iter:06d}.png")
            plt.close()

            elite = jax.tree.map(lambda x: x[0], population)
            rng, _rng = split(rng)
            state, statevid = rollout_plife(rng, elite, rollout_steps=args.rollout_steps * 3, return_statevid=True)
            vid = jax.vmap(render_fn_clip, in_axes=(0, None))(statevid, elite)
            vid = np.array((vid*255).astype(jnp.uint8))
            imageio.mimwrite(f'{args.save_dir}/vid.mp4', vid, fps=100, codec='libx264')



if __name__ == '__main__':
    main(parse_args())

