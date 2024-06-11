# import os, sys, glob, pickle
# from functools import partial

import numpy as np
# import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_theme()

from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

import jax
import jax.numpy as jnp
from jax.random import split

import flax
import flax.linen as nn
from flax.training.train_state import TrainState

import optax

import jax
from PIL import Image
import requests
from transformers import AutoProcessor, FlaxCLIPModel

# import lovely_jax as lj
# lj.monkey_patch()


def augment_image(_rng, img):
    H, W, D = img.shape
    assert H==W
    
    _rng1, _rng2 = split(_rng)
    crop_ratio = jax.random.uniform(_rng1, (2, ), minval=0.5, maxval=1.)
    crop_ratio = crop_ratio.at[1].set(crop_ratio[0])
    crop_loc = jax.random.uniform(_rng2, (2, ), minval=0., maxval=(1.-crop_ratio)*H)
    
    scale = 1./crop_ratio
    translation = -scale*crop_loc
    img = jax.image.scale_and_translate(img, (224, 224, 3), [0, 1], scale, translation, 'linear')
    return img


def main(args):
    model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dtype = jnp.float32

    # model.params = jax.tree.map(lambda x: x.astype(dtype), model.params)
    for p in jax.tree_util.tree_leaves(model.params):
        assert p.dtype==dtype

    inputs = processor(text="a photo of a cat", return_tensors="np", padding=True)
    z_text = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    z_text = z_text/jnp.linalg.norm(z_text, axis=-1, keepdims=True)

    mean, std = jnp.array(processor.image_processor.image_mean), jnp.array(processor.image_processor.image_std)
    mean, std = jax.tree_util.tree_map(lambda x: x.astype(dtype), (mean, std))
    print(mean, std)
    
    img = jnp.full((224, 224, 3), fill_value=0.5, dtype=dtype)

    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-3))
    train_state = TrainState.create(apply_fn=None, params=dict(img=img), tx=tx)

    def loss_fn(params, _rng):
        img = params['img']
    
        # img = augment_image(_rng, img)
        
        img = rearrange((img-mean)/std, 'H W D -> 1 D H W')
        z_img = model.get_image_features(img)
        z_img = z_img/jnp.linalg.norm(z_img, axis=-1, keepdims=True)
        loss = -(z_img@z_text.T).mean()
        return loss, None
    
    @jax.jit
    def train_step(train_state, _rng):
        # loss = loss_fn(train_state.params, _rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, _rng)
        assert all([p.dtype==dtype for p in jax.tree_util.tree_leaves((loss, grads))])
        # loss = grads['img'].mean()
        # train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss
    
    # loss_fn(train_state.params)

    rng = jax.random.PRNGKey(0)
    losses = []
    for i in tqdm(range(1000)):
        rng, _rng = split(rng)
        train_state, loss = train_step(train_state, _rng)
        losses.append(loss)
        # print(loss.item())


if __name__ == "__main__":
    main(None)

