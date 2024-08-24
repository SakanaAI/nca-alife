import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat, rearrange

from functools import partial

import flax.linen as nn
from einops import rearrange, reduce, repeat

class BoidNetwork(nn.Module):
    @nn.compact
    def __call__(self, x, mask): # n_neighbors, 4
        x = jnp.pad(x, pad_width=1, mode='wrap')
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = (x*mask[:, None]).mean(axis=0) # 8
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=2)(x)
        return x


# rotation and translation
def get_transformation_mats(x, v):
    (x, y), (u, v) = x, v
    global2local = jnp.array([[u, v, -u*x-v*y], [-v, u, v*x-u*y], [0, 0, 1] ])
    local2global = jnp.array([ [u, -v, x], [v, u, y], [0, 0, 1]])
    return global2local, local2global

def get_rotation_mats(v):
    u, v = v
    global2local = jnp.array([[u, v, 0], [-v, u, 0], [0, 0, 1]])
    local2global = jnp.array([[u, -v, 0], [v, u, 0], [0, 0, 1]])
    return global2local, local2global

class Boids():
    def __init__(self, n_boids=256, n_neighbors=8, search_space="init+dynamics", dt=0.01):
        self.n_boids = n_boids
        self.n_neighbors = n_neighbors
        self.search_space = search_space.split('+')
        self.dt = dt
        self.net = BoidNetwork()

    def default_params(self, rng):
        net_params = self.net.init(rng, jnp.zeros((1, 4))) # unconstrainted
        return dict(net_params=net_params)
        
    def init_state(self, rng, params):
        _rng1, _rng2, _rng3 = split(rng, 3)
        x = jax.random.uniform(_rng1, (self.n_boids, 2), minval=0., maxval=1.)
        v = jax.random.normal(_rng2, (self.n_boids, 2))
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return dict(x=x, v=v)
    
    def step_state(self, rng, state, params):
        x, v = state['x'], state['v'] # n_boids, 2

        def get_dv(xi, vi): # 2
            distance = jnp.linalg.norm(x-xi, axis=-1) # n_boids
            idx_neighbor = jnp.argsort(distance)[1:self.n_neighbors+1]
            xn, vn = x[idx_neighbor], v[idx_neighbor] # n_neighbors, 2

            g2l, l2g = get_transformation_mats(xi, vi) # 3, 3
            g2lr, l2gr = get_rotation_mats(vi) # 3, 3

            xn = jnp.concatenate([xn, jnp.ones((self.n_neighbors, 1))], axis=-1) # n_neighbors, 3
            xn = g2l @ xn[:, :, None] # n_neighbors, 3, 1
            xn = xn[:, :2, 0] # n_neighbors, 2

            vn = jnp.concatenate([vn, jnp.ones((self.n_neighbors, 1))], axis=-1) # n_neighbors, 3
            vn = g2lr @ vn[:, :, None] # n_neighbors, 3, 1
            vn = vn[:, :2, 0] # n_neighbors, 2
            
            # mask = distance[idx_neigbhor]<







            dv = jnp.mean(vn, axis=0) - vi # 2

            dv = jnp.concatenate([dv, jnp.zeros(1)], axis=0) # 3
            dv = l2gr @ dv[:, None] # 3, 1
            dv = dv[:2, 0] # 2
            return dv

        dv = jax.vmap(get_dv)(x, v)

        v = v + dv*.01
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

        x = x + v * self.dt
        x = x%1. # circular boundary
        return dict(x=x, v=v)
    
    def render_state(self, state, params, img_size=256):
        x, v = state['x'], state['v'] # n_boids, 2

        global2local, local2global = jax.vmap(get_transformation_mats)(x, v) # n_boids, 3, 3
        local_triangle_coords = jnp.array([[0, 1.], [0, -1.], [3, 0.]])/100.
        local_triangle_coords = jnp.concatenate([local_triangle_coords, jnp.ones((3, 1))], axis=-1)
        local_triangle_coords = local_triangle_coords[:, :, None] # 3, 3, 1

        global_triangle_coords = local2global[:, None, :, :] @ local_triangle_coords[None, :, :, :]
        global_triangle_coords = global_triangle_coords[:, :, :2, 0]
        img = jnp.ones((img_size, img_size, 3))

        x, y = jnp.linspace(0, 1, img_size), jnp.linspace(0, 1, img_size)
        x, y = jnp.meshgrid(x, y, indexing='ij')
        def render_triangle(img, triangle):
            # Compute barycentric coordinates
            v0 = triangle[2] - triangle[0]
            v1 = triangle[1] - triangle[0]
            v2 = jnp.stack([x, y], axis=-1) - triangle[0]
            
            d00 = jnp.dot(v0, v0)
            d01 = jnp.dot(v0, v1)
            d11 = jnp.dot(v1, v1)
            d20 = jnp.dot(v2, v0)
            d21 = jnp.dot(v2, v1)
            
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
            
            # Check if point is inside triangle
            # mask = (u >= 0) & (v >= 0) & (w >= 0)

            sharp = 50
            mask = jax.nn.sigmoid(sharp*u) * jax.nn.sigmoid(sharp*v) * jax.nn.sigmoid(sharp*w)

            mask = 1-mask.astype(jnp.float32)
            img = jnp.minimum(img, mask[..., None])
            return img, None
        img, _ = jax.lax.scan(render_triangle, img, global_triangle_coords)
        return img
    
