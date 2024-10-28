import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split


def int2binary(x, n_bits=18, bits_per_token=1):
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

class GameOfLife():
    def __init__(self, grid_size=64):
        self.grid_size = grid_size

        self.params_max = 2**18 - 1
        self.params_gol = 6152

    def default_params(self, rng):
        return jax.random.randint(rng, (), minval=0, maxval=self.params_max)
    
    def init_state(self, rng, params):
        _rng1, _rng2 = split(rng)
        sparsity = jax.random.uniform(_rng1, shape=(), minval=0.05, maxval=0.4)
        state = jax.random.uniform(_rng2, shape=(self.grid_size, self.grid_size), minval=0, maxval=1)
        state = jnp.floor(state+sparsity).astype(int)
        return state
    
    def step_state(self, rng, state, params):
        state_f = state.astype(float)
        n_neighbors = conv2d_3x3_sum(state_f) - state_f
        update_idx = state_f * 9 + n_neighbors
        next_state = int2binary(params)[update_idx.astype(int)]
        return next_state
    
    def render_state(self, state, params, img_size=None):
        img = repeat(state.astype(float), "H W -> H W 3")
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img

class GameOfLifeInit():
    def __init__(self, grid_size=64):
        self.grid_size = grid_size

        self.params_dyn = 6152
        # self.params_dyn = int2binary(self.params_gol)
        # print(self.params_dyn)

    def default_params(self, rng):
        params_init = jax.random.uniform(rng, (self.grid_size, self.grid_size), minval=0, maxval=1)
        params_init = jnp.floor(params_init+0.4).astype(int)
        return dict(params_init=params_init)
    
    def init_state(self, rng, params):
        return params['params_init']
    
    def step_state(self, rng, state, params):
        params = self.params_dyn
        state_f = state.astype(float)
        n_neighbors = conv2d_3x3_sum(state_f) - state_f
        update_idx = state_f * 9 + n_neighbors
        next_state = int2binary(params)[update_idx.astype(int)]
        return next_state
    
    def render_state(self, state, params, img_size=None):
        img = repeat(state.astype(float), "H W -> H W 3")
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img
