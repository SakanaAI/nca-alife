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

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoProcessor, FlaxCLIPModel

from evosax import CMA_ES, SimpleGA

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--dtype", type=str, default='float32')
group.add_argument("--device", type=str, default='cuda:0')

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
group.add_argument("--rollout_steps", type=int, default=64)

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


def main(args):
    plife = ParticleLife(args.n_particles, args.n_colors, n_dims=args.n_dims, dt=args.dt,
                         half_life=args.half_life, rmax=args.rmax)

    clip_model = MyFlaxCLIP(args.clip_model)

    z_text = clip_model.embed_text([args.prompt])

    print(z_text.shape)

    rng = jax.random.PRNGKey(args.seed)
    strategy = SimpleGA(popsize=args.bs, num_dims=args.n_colors*args.n_colors)
    es_params = strategy.default_params
    state = strategy.initialize(rng, es_params)

    def rollout_core(rng, env_params):
        state = plife.get_random_init_state(rng)

        def step(state, _):
            state = plife.forward_step(state, env_params)
            return state, None
        state, _ = jax.lax.scan(step, state, length=args.rollout_steps)

        render_fn = partial(plife.render_state_heavy, img_size=256, radius=0.5, sharpness=10.,
                            color_palette=color_palette)
        img = render_fn(state)
        return img, state

    env_params_default = plife.get_random_env_params(rng)
    def calc_fitness(rng, x):
        x = rearrange(x, '(K K2) -> K K2', K=args.n_colors)
        env_params = copy.copy(env_params_default)
        env_params['alphas'] = x

        img, state = rollout_core(rng, env_params)

        img = img[:224, :224]  # TODO augmentations

        z_img = clip_model.get_image_features(rearrange((img-img_mean)/img_std, 'H W C -> 1 C H W'))
        z_img = z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True)

        fitness = (z_text @ z_img.T).mean()
        return fitness

    calc_fitness(rng, jnp.zeros((args.n_colors*args.n_colors,)))

    def do_iter(state, rng):
        rng, _rng = split(rng)
        x, state = strategy.ask(_rng, state, es_params)
        fitness = jax.vmap(calc_fitness)(split(rng, x.shape[0]), x)
        state = strategy.tell(x, fitness, state, es_params)
        return state, fitness


    n_iters = 1000000
    n_iters_update = 100
    pbar = tqdm(total=n_iters)
    for i in range(n_iters//n_iters_update):
        rng, _rng = split(rng)
        state, fitness = jax.lax.scan(do_iter, state, split(_rng, n_iters_update))
        pbar.update(n_iters_update)

        if i%100==0:
            print(fitness.mean(), fitness.max())

    assert False

    rng, _rng = split(rng)
    env_params = jax.vmap(plife.get_random_env_params)(split(_rng, bs))
    print('env_params', jax.tree.map(lambda x: x.shape, env_params))

    def rollout_core(rng):
        rng, _rng = split(rng)
        env_params = plife.get_random_env_params(_rng)
        rng, _rng = split(rng)
        state = plife.get_random_init_state(_rng)
        
        def step(state, _):
            state = plife.forward_step(state, env_params)
            return state, None
            
        state, _ = jax.lax.scan(step, state, length=1000)
        return state
            
    def rollout_viz():
        pass

    rng, _rng = split(rng)
    state = jax.vmap(rollout_core)(split(_rng, bs))
    print('state', jax.tree.map(lambda x: x.shape, state))

    render_fn = partial(plife.render_state_heavy, img_size=256, radius=0.5, sharpness=10., color_palette=color_palette)
    img = jax.vmap(render_fn)(state)
    print('img', jax.tree.map(lambda x: x.shape, img))


    plt.figure(figsize=(20, 10))
    plt.imshow(rearrange(img, '(B1 B2) H W D -> (B1 H) (B2 W) D', B1=4, B2=8))
    plt.savefig('./temp/imgs.png')
    plt.close()

    plt.imsave('./temp/img_raw.png', rearrange(img, '(B1 B2) H W D -> (B1 H) (B2 W) D', B1=4, B2=8))


if __name__ == '__main__':
    main(parse_args())
    
