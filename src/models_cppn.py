
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split


class CPPNNetwork(nn.Module):
    d_dim: int = 16
    n_layers: int = 1
    activation: str = 'relu'
    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.d_dim)(x)
            x = getattr(nn, self.activation)(x)
        x = nn.Dense(features=3)(x)
        return x


class CPPN():
    def __init__(self, grid_size=64, d_dim=16, n_layers=1, activation='relu', inputs='xyrt'):
        self.grid_size = grid_size
        self.cppn = CPPNNetwork(d_dim=d_dim, n_layers=n_layers, activation=activation)
        self.inputs = inputs

    def default_params(self, rng):
        rng, _rng = split(rng)
        net_params = self.cppn.init(_rng, jnp.zeros((len(self.inputs), ))) # unconstrainted
        return dict(net_params=net_params)
    
    def render(self, params, img_size=None):
        x = jnp.linspace(-3, 3, self.grid_size)
        y = jnp.linspace(-3, 3, self.grid_size)
        x, y = jnp.meshgrid(x, y, indexing='ij') # (grid_size, grid_size)
        x, y = y, x # reverse for image coordinates
        r = jnp.sqrt(x**2 + y**2)
        theta = jnp.arctan2(y, x)
        inp = []
        if 'x' in self.inputs:
            inp.append(x)
        if 'y' in self.inputs:
            inp.append(y)
        if 'r' in self.inputs:
            inp.append(r)
        if 't' in self.inputs:
            inp.append(theta)
        inp = jnp.stack(inp, axis=-1)
        img = jax.vmap(jax.vmap(self.cppn.apply, in_axes=(None, 0)), in_axes=(None, 0))(params['net_params'], inp)
        img = jax.nn.sigmoid(img)
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img
        


