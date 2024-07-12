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

# from evosax import CMA_ES, SimpleGA

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
group.add_argument("--rollout_steps", type=int, default=1000)

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

    plife = ParticleLife(args.n_particles, args.n_colors, n_dims=args.n_dims, dt=args.dt,
                         half_life=args.half_life, rmax=args.rmax)
    clip_model = MyFlaxCLIP(args.clip_model)
    z_text = clip_model.embed_text([args.prompt])


    # render_fn_clip = partial(plife.render_state_heavy, img_size=1024, radius=2., sharpness=10.)
    # render_fn_vid = partial(plife.render_state_heavy, img_size=256, radius=0.5, sharpness=10.)


    # render_fn_clip = partial(plife.render_state_heavy, img_size=224, radius=0.4375, sharpness=3.)
    render_fn_clip = partial(plife.render_state_light, img_size=224, radius=0)

    def get_final_state(rng, env_params):
        state = plife.get_random_init_state(rng)
        def step(state, _):
            state = plife.forward_step(state, env_params)
            return state, None
        state, _ = jax.lax.scan(step, state, length=args.rollout_steps)
        return state
    
    # def get_state_video(rng, env_params):
    #     state = plife.get_random_init_state(rng)
    #     def step(state, _):
    #         state = plife.forward_step(state, env_params)
    #         return state, state
    #     state, statevid = jax.lax.scan(step, state, length=args.rollout_steps)
    #     return statevid

    bs = 32
    rng, _rng = split(rng)
    env_params = jax.vmap(plife.get_random_env_params)(split(_rng, bs))
    print(jax.tree.map(lambda x: x.shape, env_params))
    rng, _rng = split(rng)

    print('starting sim')
    state = jax.vmap(get_final_state)(split(_rng, bs), env_params)
    print('starting rendering')
    img = jax.vmap(render_fn_clip)(state)
    print('starting saving')

    z_img = jax.vmap(clip_model.embed_img)(img)
    print(z_text.shape, z_img.shape)
    scores = (z_text @ z_img.T).flatten().tolist()
    print(scores)

    plt.figure(figsize=(20, 10))
    for i in range(bs):
        plt.subplot(4, 8, i+1)
        plt.imshow(img[i])
        plt.axis('off')
        plt.title(f'{scores[i]:.2f}')
    plt.savefig('./temp/imgs_fig.png')

    img = rearrange(img, '(B1 B2) H W C -> (B1 H) (B2 W) C', B1=4)
    print(img.shape)
    print(img.max(), img.min())
    plt.imsave('./temp/imgs.png', img)


    rng, _rng = split(rng)
    env_params_default = plife.get_random_env_params(_rng)
    def mutate(rng, x):
        rng, _rng = split(rng)
        mask = (jax.random.uniform(_rng, x.shape) < 0.15).astype(x.dtype)
        noise = jax.random.uniform(rng, x.shape, minval=-1., maxval=1.)
        return (1.-mask)*x + mask*noise
        
    def calc_fitness(rng, x):
        env_params = {k: v for k, v in env_params_default.items()}
        env_params['alphas'] = rearrange(x, '(K K2) -> K K2', K=args.n_colors)
        state = get_final_state(rng, env_params)
        img = render_fn_clip(state)
        z_img = clip_model.embed_img(img)
        return (z_text @ z_img.T).mean(), img
    
    mutate = jax.jit(jax.vmap(mutate))
    calc_fitness = jax.jit(jax.vmap(calc_fitness))

    rng, _rng = split(rng)
    population = jax.random.uniform(_rng, (bs, args.n_colors*args.n_colors), minval=-1., maxval=1.)
    for i_iter in tqdm(range(args.n_iters)):
        rng, _rng = split(rng)
        fitness, img = calc_fitness(split(_rng, bs), population)
        idx = jnp.argsort(fitness, descending=True)[:bs//2]
        elite = population[idx[0]]

        rng, _rng = split(rng)
        population = population[idx[jax.random.randint(_rng, (bs,), minval=0, maxval=bs//2)]]

        rng, _rng = split(rng)
        population = mutate(split(_rng, bs), population)

        population = population.at[0].set(elite)
        
        if i_iter % 10 == 0:
            print(f'mean fitness: {fitness.mean()}, max fitness: {fitness.max()}')
        if i_iter % 100 == 0:
            fitness, img = calc_fitness(split(_rng, bs), population)
            # img = rearrange(img, '(B1 B2) H W C -> (B1 H) (B2 W) C', B1=4)
            # plt.imshow(img)

            plt.figure(figsize=(20, 10))
            for i in range(bs):
                plt.subplot(4, 8, i+1)
                plt.imshow(img[i])
                plt.axis('off')
                plt.title(f'{fitness[i]:.4f}')
            plt.tight_layout()
            plt.savefig(f'./temp/imgs_{i_iter}.png')
            plt.close()


def main_bug(args):
    rng = jax.random.PRNGKey(args.seed)

    plife = ParticleLife(args.n_particles, args.n_colors, n_dims=args.n_dims, dt=args.dt,
                         half_life=args.half_life, rmax=args.rmax)
    def get_final_state(rng, env_params):
        state = plife.get_random_init_state(rng)
        def step(state, _):
            state = plife.forward_step(state, env_params)
            return state, 1.
        state, _ = jax.lax.scan(step, state, length=args.rollout_steps, unroll=4)

        return state
    # try unroll parameter
    # try fori loop
    # 
    
    bs = 32
    rng, _rng = split(rng)
    env_params = jax.vmap(plife.get_random_env_params)(split(_rng, bs))
    print(jax.tree.map(lambda x: x.shape, env_params))
    rng, _rng = split(rng)

    fn = jax.jit(jax.vmap(get_final_state))

    for i in tqdm(range(1000)):
        # fn = jax.jit(partial(jax.vmap(get_final_state)))
        # rng, _rng = split(rng)
        fn(split(_rng, bs), env_params)

def main_jaxmd(args):
    from jax_md import partition
    from jax_md import space
    
    box_size = 1.
    cell_size = 0.1
    displacement_fn, shift_fn = space.periodic(box_size)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size, capacity_multiplier=12.5)

    rng = jax.random.PRNGKey(0)
    R = jax.random.uniform(rng, (1000, 2), minval=0., maxval=1.)
    neighbors = neighbor_list_fn.allocate(R) # Create a new neighbor list.
    print(neighbors.idx.shape)

    rng, _rng = split(rng)
    R = jax.random.uniform(rng, (1000, 2), minval=0., maxval=1.)
    neighbors = neighbors.update(R)
    print(neighbors.idx.shape)
    print(neighbors.idx)

    print('here')
    for i in range(100):
        Rb = R[neighbors.idx[:, i]]
        d = jnp.linalg.norm(R-Rb, axis=-1)
        print(d.mean())






if __name__ == '__main__':
    main_jaxmd(parse_args())



    
    
