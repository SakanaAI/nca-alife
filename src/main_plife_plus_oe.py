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
from create_sim import create_sim, rollout_and_embed_simulation, FlattenSimulationParameters
from models.models_plife_plus import ParticleLifePlus

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")
# group.add_argument("--sim", type=str, default='boids')

group = parser.add_argument_group("data")
group.add_argument("--n_rollout_imgs", type=int, default=8)
# group.add_argument("--prompts", type=str, default="an artificial cell,a bacterium")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14
group.add_argument("--coef_prompt", type=float, default=1.)
group.add_argument("--coef_softmax", type=float, default=0.)
group.add_argument("--coef_novelty", type=float, default=0.)

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1)
group.add_argument("--pop_size", type=int, default=8)
group.add_argument("--n_iters", type=int, default=1000)
group.add_argument("--sigma", type=float, default=0.1)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    # if len(args.prompts.split(",")) > 1: # doing multiple prompts
    #     args.n_rollout_imgs = len(args.prompts.split(","))
    print(args)
    
    sim = ParticleLifePlus(n_particles=1000, dt=0.02, render_radius=0.04, sharpness=30., background_color='black')
    sim = FlattenSimulationParameters(sim)
    clip_model = MyFlaxCLIP(args.clip_model)
    rollout_fn = partial(rollout_and_embed_simulation, sim=sim, clip_model=clip_model, rollout_steps=2048, n_rollout_imgs=args.n_rollout_imgs, chunk_ends=True)
    rng = jax.random.PRNGKey(args.seed)

    # z_text = clip_model.embed_text(args.prompts.split(",")) # P D

    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=sim.n_params, sigma_init=args.sigma)
    es_params = strategy.default_params
    es_params = es_params.replace(init_min=-1., init_max=1.)
    rng, _rng = split(rng)
    es_state = strategy.initialize(_rng, es_params)

    def calc_loss(rng, params):
        rollout_data = rollout_fn(rng, params)
        z = rollout_data['z']
        scores_novelty = (z @ z.T) # T T
        scores_novelty = jnp.tril(scores_novelty, k=-1)
        loss_novelty = scores_novelty.max(axis=-1).mean() if args.n_rollout_imgs > 1 else 0.
        loss_dict = dict(loss_novelty=loss_novelty)
        return loss_novelty, loss_dict

    @jax.jit
    def do_iter(es_state, rng):
        rng, _rng = split(rng)
        params, next_es_state = strategy.ask(_rng, es_state, es_params)
        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))
        rng, _rng = split(rng)
        loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params)
        loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict))  # mean over bs
        next_es_state = strategy.tell(params, loss, next_es_state, es_params)
        data = dict(best_loss=next_es_state.best_fitness, generation_loss=loss.mean(), loss_dict=loss_dict)
        return next_es_state, data

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        es_state, di = do_iter(es_state, _rng)

        data.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())
        print(es_state.best_fitness.item())
        if args.save_dir is not None and (i_iter % (args.n_iters//5)==0 or i_iter==args.n_iters-1):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            best = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best)

if __name__ == '__main__':
    main(parse_args())
