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
group.add_argument("--n_particles", type=int, default=4000)
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
group.add_argument("--rollout_steps", type=int, default=2000)

group.add_argument("--bs", type=int, default=32)
group.add_argument("--n_iters", type=int, default=32)


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
    # render_fn = partial(plife.render_state_heavy, img_size=224, radius=8e-3, sharpness=5e3)
    render_fn = partial(plife.render_state_heavy, img_size=224, radius=1.5e-2, sharpness=3e3)

    def rollout_plife(rng, env_params, rollout_steps=args.rollout_steps, return_statevid=False):
        state = plife.get_random_init_state(rng)
        def step(state, _):
            next_state = plife.forward_step(state, env_params)
            return next_state, (state if return_statevid else None)
        state, statevid = jax.lax.scan(step, state, length=rollout_steps)
        return state, statevid
    
    def gen_random_individual(rng):
        a = plife.get_default_env_params()
        a['alpha'] = jax.random.uniform(rng, a['alpha'].shape, minval=-1, maxval=1)
        return a

    rng, _rng = split(rng)
    population = jax.vmap(gen_random_individual)(split(_rng, args.bs))

    @jax.jit
    def do_iter(rng):
        state, _ = jax.vmap(rollout_plife, in_axes=(None, 0))(rng, population)
        img = jax.vmap(render_fn)(state, population)
        z_img = jax.vmap(clip_model.embed_img)(img)
        return img, z_img

    img, z_img = [], []
    for _rng in tqdm(split(rng, args.n_iters)):
        imgi, z_imgi = do_iter(_rng)
        img.append(imgi)
        z_img.append(z_imgi)
    img = jnp.stack(img, axis=1)
    z_img = jnp.stack(z_img, axis=1)  # E S D

    if args.save_dir is not None:
        plt.imsave(f"{args.save_dir}/img.png", rearrange(img[:8, :8], "b1 b2 h w c -> (b1 h) (b2 w) c"))
    
    A = rearrange(z_img, "E S D -> E S D") @ rearrange(z_img, "E S D -> E D S") # E S S
    B = rearrange(z_img, "E S D -> S E D") @ rearrange(z_img, "E S D -> S D E") # S E E

    print(f"Avg Similarity over  envs: {B.mean().item(): 0.5f}", )
    print(f"Avg Similarity over seeds: {A.mean().item(): 0.5f}", )

    prompts = ["a cell", "a diverse ecosystem of cells moving around", "a biological cell under the microscope", "mitosis",
               "a caterpillar", "a colorful caterpillar",
               "bacteria", "bacteria under the microscope",
               "particle life", "an artificial life simulation",
               "a black screen", "nothing",
               "outer space", "milky way in the night sky",
               "self-replicating organisms", "petri dish"]
    
    for prompt in prompts:
        print("----------------------------------------------")
        print(f"Prompt: {prompt}")
        z_text = clip_model.embed_text([prompt])[0] # D
        scores = z_img @ z_text # E S

        print(f"Std over  envs: {scores.std(axis=0).mean().item(): 0.5f}", )
        print(f"Std over seeds: {scores.std(axis=1).mean().item(): 0.5f}", )

        scores = scores.mean(axis=-1)
        print(f"Avg Env Score: {scores.mean().item(): 0.5f}")
        print(f"Std Env Score: {scores.std().item(): 0.5f}")
        print(f"Min Env Score: {scores.min().item(): 0.5f}")
        print(f"Max Env Score: {scores.max().item(): 0.5f}")



if __name__ == '__main__':
    main(parse_args())

