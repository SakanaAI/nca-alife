import argparse
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from tqdm.auto import tqdm
from einops import rearrange, repeat

import util
from clip_jax import MyFlaxCLIP
clip_model = MyFlaxCLIP()

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)

group = parser.add_argument_group("data")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=512)
group.add_argument("--grid_size", type=int, default=64)
group.add_argument("--rollout_steps", type=int, default=4096)
group.add_argument("--n_steps_clip", type=int, default=32) # number of clip ckpts within rollouts

group.add_argument("--start", type=int, default=0) # start range for params search
group.add_argument("--end", type=int, default=262144) # end range for params search

args = parser.parse_args()

resize_fn = partial(jax.image.resize, shape=(224, 224), method='nearest')

def int2binary(x, n_bits=32, bits_per_token=1):
    bit_positions = 2 ** jnp.arange(n_bits)
    binary = jnp.bitwise_and(x, bit_positions) > 0
    tokens = binary.reshape(-1, bits_per_token)
    tokens = (tokens * (2 ** jnp.arange(bits_per_token))).sum(axis=-1)
    return tokens

def conv2d_3x3_sum(x):
    x_padded = jnp.pad(x, pad_width=1, mode='wrap')
    kernel = jnp.ones((3, 3))
    return jax.lax.conv_general_dilated(
            x_padded[None, None, :, :],  # Add batch and channel dimensions
            kernel[None, None, :, :],  # Add input and output channel dimensions
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))[0, 0]

def step_ca(state, _, params):
    state_f = state.astype(float)
    n_neighbors = conv2d_3x3_sum(state_f) - state_f
    update_idx = state_f * 9 + n_neighbors
    next_state = params[update_idx.astype(int)]
    return next_state, None

# def unroll_ca(state_init, params, n_steps=512):
#     state_final, state_vid = jax.lax.scan(partial(step_ca, params=params), state_init, None, length=n_steps)
#     return state_vid

def unroll_ca(state_init, params):
    def chunk_step(state_init, _):
        state, _ = jax.lax.scan(partial(step_ca, params=params), state_init, None, length=args.rollout_steps//args.n_steps_clip)
        return state, state_init
    _, state_vid = jax.lax.scan(chunk_step, state_init, None, length=args.n_steps_clip)
    return state_vid

params_gol = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
x_gol_glider = jnp.zeros((32, 32), dtype=int)
x_gol_glider = x_gol_glider.at[:3, :3].set(jnp.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]))

rng = jax.random.PRNGKey(args.seed)
sparsity = jnp.linspace(.05, .4, num=args.bs)
state_init = jax.random.uniform(rng, shape=(args.bs, args.grid_size, args.grid_size), minval=0, maxval=1)
state_init = jnp.floor(state_init+sparsity[:, None, None]).astype(int)

@jax.jit
def eval_clip_complexity(params):
    state_vid = jax.vmap(partial(unroll_ca, params=params))(state_init) # B T H W
    print('state_vid ', state_vid.shape)
    vid_clip = jax.vmap(jax.vmap(resize_fn))(state_vid.astype(jnp.float32))
    vid_clip = repeat(vid_clip, "... -> ... 3")
    print('clip vid ', vid_clip.shape)
    z_img = jax.vmap(jax.vmap(clip_model.embed_img))(vid_clip)
    scores = (z_img@z_img.mT)
    print('scores ', scores.shape)
    scores_tril = jax.vmap(partial(jnp.tril, k=-1))(scores)
    similarities = (scores_tril).max(axis=-1)
    return dict(state_vid=state_vid, z_img=z_img, scores=scores, similarities=similarities)

all_params = jax.vmap(partial(int2binary, n_bits=18))(jnp.arange(args.start, args.end))
print(all_params.shape, all_params.dtype, all_params.min(), all_params.max())

data = []
for i in tqdm(range(len(all_params))):
    params = all_params[i]
    di = eval_clip_complexity(params)

    data.append(dict(scores=di['scores'].mean(axis=0), similarities=di['similarities'].mean(axis=0)))

data = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *data)

if args.save_dir is not None:
    data = jax.tree.map(lambda x: np.array(x), data)
    all_params = jax.tree.map(lambda x: np.array(x), all_params)
    util.save_pkl(args.save_dir, "all_params", all_params)
    util.save_pkl(args.save_dir, "data", data)
