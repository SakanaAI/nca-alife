
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split

import matplotlib.colors as mcolors


class DNCANetwork(nn.Module):
    d_state: int = 16
    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, pad_width=1, mode='wrap')
        x = nn.Conv(features=4, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.Conv(features=16, kernel_size=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.d_state, kernel_size=(1, 1))(x)
        return x


class DNCA():
    def __init__(self, grid_size=64, d_state=8, n_groups=1, identity_bias=0., temperature=1., color_map='fixed'):
        self.grid_size = grid_size
        self.d_state, self.n_groups = d_state, n_groups
        self.dnca = DNCANetwork(d_state=d_state*n_groups)

        self.identity_bias, self.temperature = identity_bias, temperature

        self.color_map = color_map
        self.color_palette = 'ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00'
        self.color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in self.color_palette.split('-')]) # 8 3

    def default_params(self, rng):
        params = dict()
        if self.color_map != 'fixed':
            rng, _rng = split(rng)
            params['color_map'] = jax.random.normal(_rng, (self.n_groups, self.d_state, 3)) # unconstrainted

        rng, _rng = split(rng)
        params['net_params'] = self.dnca.init(_rng, jnp.zeros((self.grid_size, self.grid_size, self.d_state*self.n_groups))) # unconstrainted

        rng, _rng = split(rng)
        params['init'] = jax.random.normal(_rng, (self.n_groups, self.d_state)) # unconstrainted
        return params
    
    def init_state(self, rng, params):
        init = repeat(params['init'], "G D -> H W G D", H=self.grid_size, W=self.grid_size)
        state = jax.random.categorical(rng, init, axis=-1)
        return state
    
    def step_state(self, rng, state, params):
        state_oh = jax.nn.one_hot(state, self.d_state)
        state_oh_f = rearrange(state_oh, "H W G D -> H W (G D)")
        logits = self.dnca.apply(params['net_params'], state_oh_f)
        logits = rearrange(logits, "H W (G D) -> H W G D", G=self.n_groups)
        
        # identity_bias = jax.nn.sigmoid(params['identity_bias'])*10
        next_state = jax.random.categorical(rng, (logits + state_oh*self.identity_bias)/self.temperature, axis=-1)
        return next_state
    
    def render_state(self, state, params, img_size=None):
        if self.color_map=='fixed':
            img = self.color_palette[state[:, :, 0]]
        else:
            def get_color(color_map, state):
                return color_map[state]
            # color_map: G D 3 # state: H W G
            get_color = jax.vmap(get_color, in_axes=(0, 2))
            img = get_color(jax.nn.sigmoid(params['color_map']), state)
            img = img.mean(axis=0) # average over groups

        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img

