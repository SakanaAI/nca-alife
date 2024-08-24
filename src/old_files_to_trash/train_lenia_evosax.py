import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['JAX_DEBUG_NANS'] = 'True'
# os.environ['JAX_ENABLE_X64'] = 'True'

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import numpy as np

from tqdm.auto import tqdm
import copy
from einops import rearrange, reduce, repeat

import jax
import jax.numpy as jnp
from jax.random import split
import optax
from flax.training.train_state import TrainState

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

import evosax

from lenia import ConfigLenia, Lenia
from clip_jax import MyFlaxCLIP

import evosax

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")

group = parser.add_argument_group("data")
group.add_argument("--prompt", type=str, default="a biological cell")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group.add_argument("--coef_prompt", type=float, default=1.)

group = parser.add_argument_group("optimization")
group.add_argument("--algo", type=str, default="Sep_CMA_ES")
group.add_argument("--sigma", type=float, default=1.)
group.add_argument("--rollout_steps", type=int, default=200)
group.add_argument("--bs", type=int, default=32)
group.add_argument("--n_iters", type=int, default=10000)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    print('starting main')
    config_lenia = ConfigLenia(pattern_id='5N7KKM')
    print('creating lenia')
    lenia = Lenia(config_lenia)
    print('creating clip')
    clip_model = MyFlaxCLIP()
    resize_fn = partial(jax.image.resize, shape=(224, 224, 3), method='nearest')

    print('embedding text')
    z_text = clip_model.embed_text([args.prompt]) # 1 D
    init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

    def unroll_geno(geno, center_phenotype=False):
        geno = jax.nn.sigmoid(geno)  # NOTE: I AM DOING SIGMOID
        carry = lenia.express_genotype(init_carry, geno)
        lenia_step = partial(lenia.step, phenotype_size=config_lenia.world_size, center_phenotype=center_phenotype, record_phenotype=True)
        carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(args.rollout_steps)) # changed from lenia._config.n_step
        return accum.phenotype

    def loss_fn(geno):
        pheno = unroll_geno(geno, center_phenotype=True) # T H W D
        img = pheno[-1, 40:88, 40:88, :] # H W D
        z_img = clip_model.embed_img(resize_fn(img)) # D

        loss_dict = {}
        loss_dict['loss_prompt'] = -rearrange(z_img, "D -> 1 D") @ rearrange(z_text, "1 D -> D 1") # 1 1
        loss = 0.
        for k in loss_dict:
            loss_dict[k] = loss_dict[k].mean()
            coef = getattr(args, k.replace('loss_', 'coef_'))
            loss = loss + coef * loss_dict[k].mean()
        loss_dict['loss'] = loss
        return loss, loss_dict
    
    print('creating seed')
    rng = jax.random.PRNGKey(args.seed)

    # genos = repeat(init_genotype, "D -> B D", B=args.bs)
    # genos = jax.random.normal(rng, (args.bs, *init_genotype.shape))
    # genos = jax.nn.sigmoid(genos)

    print('creating strategy')
    strategy = evosax.Strategies[args.algo](popsize=args.bs, num_dims=45+32*32*3, sigma_init=args.sigma)
    # es_params = strategy.default_params.replace(init_min=-5, init_max=5)
    es_params = strategy.default_params.replace()

    print('creating state')
    rng, _rng = split(rng)
    state = strategy.initialize(_rng, es_params)

    def inv_sigmoid(x):
        return jnp.log(x) - jnp.log1p(-x)

    # NOTE: change the mean to the initial genotype
    state = state.replace(mean=inv_sigmoid(init_genotype.clip(1e-6, 1.-1e-6)))

    @jax.jit
    def do_iter(state, rng):
        x, state = strategy.ask(rng, state, es_params)
        losses, di = jax.vmap(loss_fn)(x)
        state = strategy.tell(x, losses, state, es_params)
        di = jax.tree.map(lambda x: x.mean(), di)
        return state, di

    print('starting training')
    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        state, di = do_iter(state, _rng)

        pbar.set_postfix(loss=state.best_fitness.item())
        data.append({**di, 'best_fitness': state.best_fitness})

        if args.save_dir is not None and i_iter % (args.n_iters//10) == 0:
            geno = state.best_member
            pheno = unroll_geno(geno, center_phenotype=False)  # T H W D
            vid = np.array((pheno*255).astype(jnp.uint8))
            util.save_pkl(args.save_dir, 'vid', vid)
            imageio.mimwrite(f'{args.save_dir}/vid.mp4', vid, fps=20, codec='libx264')
            imageio.mimwrite(f'{args.save_dir}/vid.gif', vid, fps=20)

            pheno = unroll_geno(geno, center_phenotype=True)  # T H W D
            img = pheno[-1, 40:88, 40:88, :] # H W D
            plt.imsave(f'{args.save_dir}/img_{i_iter}.png', img)

            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, 'data', data_save)
            util.save_pkl(args.save_dir, 'state', jax.tree.map(lambda x: np.array(x), state))

if __name__ == '__main__':
    main(parse_args())

