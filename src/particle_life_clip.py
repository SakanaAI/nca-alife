import numpy as np

from tqdm.auto import tqdm
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


from particle_life_jax import ParticleLife

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__=='__main__':
    import numpy as np
    from tqdm.auto import tqdm
    plife = ParticleLife(10000, 8, n_dims=2, dt=0.001, half_life=0.04, rmax=0.1)

    bs = 32
    
    rng = jax.random.PRNGKey(0)
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

    color_palette = 'ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00'
    # render_fn = partial(plife.render_state_heavy, img_size=1024, radius=2., sharpness=10., color_palette=color_palette)
    render_fn = partial(plife.render_state_heavy, img_size=256, radius=0.5, sharpness=10., color_palette=color_palette)
    img = jax.vmap(render_fn)(state)
    print('img', jax.tree.map(lambda x: x.shape, img))


    plt.figure(figsize=(20, 10))
    plt.imshow(rearrange(img, '(B1 B2) H W D -> (B1 H) (B2 W) D', B1=4, B2=8))
    plt.savefig('./temp/imgs.png')
    plt.close()

    plt.imsave('./temp/img_raw.png', rearrange(img, '(B1 B2) H W D -> (B1 H) (B2 W) D', B1=4, B2=8))
    
    
