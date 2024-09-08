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
from create_sim import create_sim

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")
group.add_argument("--sim", type=str, default='boids')

group = parser.add_argument_group("data")
group.add_argument("--n_rollout_imgs", type=int, default=1)
group.add_argument("--prompts", type=str, default="an artificial cell,a bacterium")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14
group.add_argument("--coef_prompts", type=float, default=1.)
group.add_argument("--coef_novelty", type=float, default=0.)

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1)
group.add_argument("--pop_size", type=int, default=16)
group.add_argument("--n_iters", type=int, default=10000)
group.add_argument("--sigma", type=float, default=1.)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    sim = create_sim(args.sim)
    clip_model = MyFlaxCLIP(args.clip_model)
    z_text = clip_model.embed_text(args.prompts.split(",")) # P D

    rng = jax.random.PRNGKey(args.seed)
    param_reshaper = evosax.ParameterReshaper(sim.default_params(rng))
    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=param_reshaper.total_params, sigma_init=args.sigma)
    es_params = strategy.default_params

    rng, _rng = split(rng)
    es_state = strategy.initialize(_rng, es_params)

    def calc_loss(rng, params):
        def step(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, state
        state_init = sim.init_state(rng, params)
        state_final, state_vid = jax.lax.scan(step, state_init, split(rng, sim.rollout_steps))

        sr = sim.rollout_steps//args.n_rollout_imgs
        idx_downsample = jnp.arange(sr-1, sim.rollout_steps, sr)
        state_vid = jax.tree.map(lambda x: x[idx_downsample], state_vid) # downsample
        vid = jax.vmap(partial(sim.render_state, params=params, img_size=224))(state_vid) # T H W C
        z_img = jax.vmap(clip_model.embed_img)(vid) # T D

        scores_novelty = (z_img @ z_img.T) # T T
        scores_novelty = jnp.tril(scores_novelty, k=-1)
        loss_novelty = scores_novelty[1:, :].max(axis=-1).mean() if args.n_rollout_imgs > 1 else 0.

        scores = z_text @ z_img.T # P T
        loss_prompts = -scores.max(axis=-1).mean()

        loss = loss_prompts * args.coef_prompts + loss_novelty * args.coef_novelty
        loss_dict = dict(loss=loss, loss_prompts=loss_prompts, loss_novelty=loss_novelty)
        return loss, loss_dict

    @jax.jit
    def do_iter(es_state, rng):
        rng, _rng = split(rng)
        x, next_es_state = strategy.ask(_rng, es_state, es_params)
        params = param_reshaper.reshape(x)
        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))
        rng, _rng = split(rng)
        loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params)
        loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict))  # mean over bs
        next_es_state = strategy.tell(x, loss, next_es_state, es_params)
        data = dict(best_loss=next_es_state.best_fitness, generation_loss=loss.mean(), loss_dict=loss_dict)
        return next_es_state, data

    @jax.jit
    def inference_video(rng, params):
        def step(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, state
        state_init = sim.init_state(rng, params)
        state_final, state_vid = jax.lax.scan(step, state_init, split(rng, int(sim.rollout_steps*1.5)))
        vid = jax.vmap(partial(sim.render_state, params=params, img_size=256))(state_vid) # T H W C
        return vid

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        es_state, di = do_iter(es_state, _rng)

        data.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())
        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            best = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best)

            params = param_reshaper.reshape_single(es_state.best_member)
            vid = inference_video(rng, params)
            vid = np.array((vid*255).astype(jnp.uint8))
            util.save_pkl(args.save_dir, "vid", vid)
            # imageio.mimwrite(f'{args.save_dir}/vid.mp4', vid, fps=30, codec='libx264')

            plt.figure(figsize=(10, 5))
            plt.subplot(211)
            plt.plot(data_save['best_loss'], color='green', label='best loss')
            plt.plot(data_save['generation_loss'], color='lightgreen', label='generation loss')
            plt.axhline(data_save['generation_loss'][0], color='r', linestyle='dashed', label='initial loss')
            plt.legend()
            plt.subplot(212); plt.imshow(rearrange(vid[::(vid.shape[0]//8), :, :, :], "T H W D -> (H) (T W) D"))
            plt.savefig(f'{args.save_dir}/overview_{i_iter:06d}.png')
            plt.close()

if __name__ == '__main__':
    main(parse_args())
