import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['JAX_DEBUG_NANS'] = 'True'
# os.environ['JAX_ENABLE_X64'] = 'True'

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


from lenia import ConfigLenia, Lenia
from clip_jax import MyFlaxCLIPBackprop


parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("model")
group.add_argument("--rmax", type=float, default=0.1)

group = parser.add_argument_group("data")
group.add_argument("--prompt", type=str, default="a biological cell")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group.add_argument("--coef_prompt", type=float, default=0.)
group.add_argument("--coef_time", type=float, default=1.)
group.add_argument("--coef_batch", type=float, default=0.)

group = parser.add_argument_group("optimization")
group.add_argument("--rollout_steps", type=int, default=1)
group.add_argument("--bs", type=int, default=8)
group.add_argument("--lr", type=float, default=3e-4)
group.add_argument("--n_iters", type=int, default=100000)
group.add_argument("--clip_grad_norm", type=float, default=1.)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args


def main(args):
    config_lenia = ConfigLenia()
    lenia = Lenia(config_lenia)
    clip_model = MyFlaxCLIPBackprop()
    lenia_step = partial(lenia.step, phenotype_size=config_lenia.world_size, center_phenotype=True, record_phenotype=True)
    resize_fn = partial(jax.image.resize, shape=(224, 224, 3), method='nearest')

    z_text = clip_model.embed_text([args.prompt]) # 1 D

    init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

    def unroll_geno(geno):
        # geno = jax.nn.sigmoid(geno)  # NOTE: I AM DOING SIGMOID
        carry = lenia.express_genotype(init_carry, geno)
        carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(args.rollout_steps)) # changed from lenia._config.n_step
        return accum.phenotype

    def loss_fn(genos):
        phenos = jax.vmap(unroll_geno)(genos)
        imgs = phenos[:, ::20, 40:88, 40:88]
        imgs = jax.vmap(jax.vmap(resize_fn))(imgs)
        z_img = jax.vmap(jax.vmap(clip_model.embed_img))(imgs) # B T D

        loss_dict = {}
        loss_dict['loss_time'] = -rearrange(z_img, "B T D -> B T D") @ rearrange(z_img, "B T D -> B D T") # B T T
        loss_dict['loss_batch'] = rearrange(z_img, "B T D -> T B D") @ rearrange(z_img, "B T D -> T D B") # T B B
        loss_dict['loss_prompt'] = -rearrange(z_img, "B T D -> B T D") @ rearrange(z_text, "1 D -> D 1") # B T 1

        loss = 0.
        for k in loss_dict:
            loss_dict[k] = loss_dict[k].mean()
            coef = getattr(args, k.replace('loss_', 'coef_'))
            loss = loss + coef * loss_dict[k].mean()
        loss_dict['loss'] = loss

        # return loss, loss_dict

        loss = phenos[:, 0].mean()
        return loss, dict(loss=loss, phenos=phenos)
    
    # @jax.jit
    def do_iter(train_state, _):
        # _ = loss_fn(train_state.params)
        grads, data = jax.grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, data

    rng = jax.random.PRNGKey(args.seed)

    # genos = repeat(init_genotype, "D -> B D", B=args.bs)
    genos = jax.random.normal(rng, (args.bs, *init_genotype.shape))
    genos = jax.nn.sigmoid(genos)
    print("DTYPE")
    print(genos.dtype)

    tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), optax.adamw(args.lr, weight_decay=0., eps=1e-8))
    train_state = TrainState.create(apply_fn=None, params=genos, tx=tx)

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        train_state, di = do_iter(train_state, None)

        pbar.set_postfix(loss=di['loss'].item())
        data.append(di)

        if args.save_dir is not None and i_iter % (args.n_iters//10) == 0:
            genos = train_state.params
            phenos = jax.vmap(unroll_geno)(genos)
            imgs = phenos[:, ::20, 40:88, 40:88]
            imgs = jax.vmap(jax.vmap(resize_fn))(imgs)
            plt.figure(figsize=(20, 20))
            plt.imshow(rearrange(imgs, "R C H W D -> (R H) (C W) D"))
            plt.ylabel("Batch"); plt.xlabel("Time")
            plt.savefig(f"{args.save_dir}/img_{i_iter}.png")
            plt.close()

            # save video now
        print(di['phenos'].min(), di['phenos'].max())

if __name__ == '__main__':
    main(parse_args())

