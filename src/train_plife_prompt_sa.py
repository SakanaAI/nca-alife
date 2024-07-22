import copy
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

group = parser.add_argument_group("model")
group.add_argument("--n_particles", type=int, default=5000)
group.add_argument("--n_colors", type=int, default=4)
group.add_argument("--n_dims", type=int, default=2)
group.add_argument("--x_dist_bins", type=int, default=7)

group = parser.add_argument_group("data")
group.add_argument("--prompt", type=str, default="an artificial cell")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group.add_argument("--render_radius", type=float, default=5e-3)
group.add_argument("--render_sharpness", type=float, default=20.)

group = parser.add_argument_group("optimization")
group.add_argument("--rollout_steps", type=int, default=1000)

group.add_argument("--bs", type=int, default=4)
group.add_argument("--n_iters", type=int, default=128)

group.add_argument("--mr", type=float, default=1e-2)
group.add_argument("--anneal_prob_end", type=float, default=1e-4)
group.add_argument("--mutate_params", type=str, default="alpha") # "alpha+beta"

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    rng = jax.random.PRNGKey(args.seed)

    plife = ParticleLife(args.n_particles, args.n_colors, n_dims=args.n_dims, x_dist_bins=args.x_dist_bins)
    clip_model = MyFlaxCLIP(args.clip_model)
    z_text = clip_model.embed_text([args.prompt])
    render_fn = partial(plife.render_state_heavy, img_size=224, radius=args.render_radius, sharpness=args.render_sharpness)
    render_fn_vid = partial(plife.render_state_heavy, img_size=512, radius=args.render_radius, sharpness=args.render_sharpness)
    render_fn_large = partial(plife.render_state_heavy, img_size=1024, radius=args.render_radius, sharpness=args.render_sharpness)

    def rollout_plife(rng, env_params, rollout_steps=args.rollout_steps, return_statevid=False):
        state_init = plife.get_init_state(rng, env_params)
        def step(state, _):
            next_state = plife.forward_step(state, env_params)
            return next_state, (state if return_statevid else None)
        state_final, state_vid = jax.lax.scan(step, state_init, length=rollout_steps)
        return state_init, state_final, state_vid

    def random_individual(rng):
        x = plife.get_default_env_params()
        if 'beta' in args.mutate_params:
            rng, _rng = split(rng)
            x['beta'] = jax.random.uniform(_rng, x['beta'].shape, minval=0., maxval=1.)
        if 'alpha' in args.mutate_params:
            rng, _rng = split(rng)
            x['alpha'] = jax.random.normal(_rng, x['alpha'].shape)
        if 'mass' in args.mutate_params:
            rng, _rng = split(rng)
            x['mass'] = 10.**jax.random.uniform(_rng, x['mass'].shape, minval=-1.5, maxval=0.) # .03 to 1
        if 'dt' in args.mutate_params:
            rng, _rng = split(rng)
            x['dt'] = 10.**jax.random.uniform(_rng, x['dt'].shape, minval=-4., maxval=-1.)
        if 'half_life' in args.mutate_params:
            rng, _rng = split(rng)
            x['half_life'] = 10.**jax.random.uniform(_rng, x['half_life'].shape, minval=-3., maxval=0.)
        if 'rmax' in args.mutate_params:
            rng, _rng = split(rng)
            x['rmax'] = 10.**jax.random.uniform(_rng, x['rmax'].shape, minval=-2., maxval=0.)
        if 'c_dist' in args.mutate_params:
            rng, _rng = split(rng)
            x['c_dist'] = jax.random.normal(_rng, x['c_dist'].shape)
        if 'x_dist' in args.mutate_params:
            rng, _rng = split(rng)
            x['x_dist'] = jax.random.normal(_rng, x['x_dist'].shape)
        return x

    def mutate(rng, x, mr=args.mr):
        x = {k: v for k, v in x.items()}
        rng, _rng = split(rng)
        xp = random_individual(_rng)
        for k in x:
            if k in args.mutate_params:
                rng, _rng = split(rng)
                mask = (jax.random.uniform(_rng, x[k].shape) < mr).astype(jnp.float32)
                x[k] = xp[k]*mask + x[k]*(1.-mask)
        return x
        
    def calc_fitness(rng, x):
        env_params = x
        state_init, state_final, state_vid = rollout_plife(rng, env_params, return_statevid=False)
        img_init = render_fn(state_init, env_params)
        img_final = render_fn(state_final, env_params)
        z_init = clip_model.embed_img(img_init)
        z_final = clip_model.embed_img(img_final)
        score = (z_text @ z_final.T).mean()
        return dict(env_params=env_params, state_init=state_init, state_final=state_final, z_init=z_init, z_final=z_final, score=score)

    # @jax.jit
    def init_iter(rng):
        _rng1, _rng2 = split(rng)
        x = random_individual(_rng1)
        m = jax.vmap(calc_fitness, in_axes=(0, None))(split(_rng2, args.bs), x)
        return (x, m)

    @jax.jit
    def do_iter(rng, i_iter, x_m):
        _rng1, _rng2, _rng3 = split(rng, 3)

        x, m = x_m
        xp = mutate(_rng1, x)
        mp = jax.vmap(calc_fitness, in_axes=(0, None))(split(_rng2, args.bs), xp)

        cond1 = mp['score'].mean() > m['score'].mean()
        cond2 = jax.random.uniform(_rng3) < (args.anneal_prob_end ** (i_iter/args.n_iters))
        cond = jnp.logical_or(cond1, cond2)
        x_m = jax.tree.map(lambda a, b: jax.lax.select(cond, a, b), (xp, mp), (x, m))
        return x_m, (xp, mp)

    data_dense, data_sparse = [], []

    rng, _rng = split(rng)
    carry = init_iter(rng)
    for i_iter in tqdm(range(args.n_iters)):
        rng, _rng = split(rng)
        carry, (_, metrics) = do_iter(_rng, i_iter, carry)

        data_dense.append(metrics)
        if i_iter % (args.n_iters//32)==0:
            metrics = copy.copy(metrics)
            metrics['img_init_clip'] = jax.vmap(render_fn)(metrics['state_init'], metrics['env_params']) # vmap over seeds
            metrics['img_final_clip'] = jax.vmap(render_fn)(metrics['state_final'], metrics['env_params'])
            metrics['img_init'] = jax.vmap(render_fn_large)(metrics['state_init'], metrics['env_params'])
            metrics['img_final'] = jax.vmap(render_fn_large)(metrics['state_final'], metrics['env_params'])
            data_sparse.append(metrics)
    
    if args.save_dir is not None:
        util.save_pkl(args.save_dir, "data_dense", data_dense)
        util.save_pkl(args.save_dir, "data_sparse", data_sparse)

        # render video
        env_params = carry[0]
        _, _, statevid = rollout_plife(rng, env_params, rollout_steps=args.rollout_steps, return_statevid=True)
        # print('statevid ', jax.tree.map(lambda x: x.shape, statevid))
        # print('env_params ', jax.tree.map(lambda x: x.shape, env_params))

        vid = jax.vmap(render_fn_vid, in_axes=(0, None))(statevid, env_params)
        vid = np.array((vid*255).astype(jnp.uint8))
        # print(vid.shape, vid.size, vid.dtype, vid.min(), vid.max())
        imageio.mimwrite(f'{args.save_dir}/vid.mp4', vid, fps=100, codec='libx264')
        


if __name__ == '__main__':
    main(parse_args())

