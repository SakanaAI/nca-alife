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
    def __call__(self, x, mask): # n_nbrs, 4
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)

        x = (x*mask[:, None]).sum(axis=0) / mask.sum() # 8

        x = nn.Dense(features=8)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        x = jax.nn.tanh(x)
        return jax.lax.select(mask.sum()>0, x, jnp.zeros_like(x))


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
    def __init__(self, n_boids=64, n_nbrs=16, visual_range=0.1, speed=0.5,
                 controller='network',
                 dt=0.01, bird_render_size=0.02, bird_render_sharpness=20, space_size=1., red_boid=True):
        self.n_boids = n_boids
        self.n_nbrs = n_nbrs
        self.visual_range = visual_range
        self.speed = speed
        self.controller = controller
        self.dt = dt
        self.bird_render_size = bird_render_size
        self.bird_render_sharpness = bird_render_sharpness
        self.space_size = space_size
        self.red_boid = red_boid

        self.net = BoidNetwork()

        assert controller=='network', 'only network controller supported for now'

    def default_params(self, rng):
        if self.controller == 'network':
            net_params = self.net.init(rng, jnp.zeros((self.n_nbrs, 4)), jnp.ones((self.n_nbrs,))) # unconstrainted
            return dict(net_params=net_params)
        else:
            coef_cohesion = 0.005
            coef_avoidance = 0.05
            coef_alignment = 0.05
            return dict(coef_cohesion=coef_cohesion, coef_avoidance=coef_avoidance, coef_alignment=coef_alignment)
        
    def init_state(self, rng, params):
        _rng1, _rng2, _rng3 = split(rng, 3)
        x = jax.random.uniform(_rng1, (self.n_boids, 2), minval=0., maxval=self.space_size)
        v = jax.random.normal(_rng2, (self.n_boids, 2))
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return dict(x=x, v=v)
    
    def _step_state_network(self, rng, state, params):
        x, v = state['x'], state['v'] # n_boids, 2

        def get_dv(xi, vi): # 2
            distance = jnp.linalg.norm(x-xi, axis=-1) # n_boids
            idx_neighbor = jnp.argsort(distance)[1:self.n_nbrs+1]
            xn, vn = x[idx_neighbor], v[idx_neighbor] # n_nbrs, 2

            g2l, l2g = get_transformation_mats(xi, vi) # 3, 3
            g2lr, l2gr = get_rotation_mats(vi) # 3, 3

            xn = jnp.concatenate([xn, jnp.ones((self.n_nbrs, 1))], axis=-1) # n_nbrs, 3
            xn = g2l @ xn[:, :, None] # n_nbrs, 3, 1
            xn = xn[:, :2, 0] # n_nbrs, 2

            vn = jnp.concatenate([vn, jnp.ones((self.n_nbrs, 1))], axis=-1) # n_nbrs, 3
            vn = g2lr @ vn[:, :, None] # n_nbrs, 3, 1
            vn = vn[:, :2, 0] # n_nbrs, 2

            inputs = jnp.concatenate([50*xn, vn], axis=-1) # n_nbrs, 4
            mask = distance[idx_neighbor] < self.visual_range
            outputs = self.net.apply(params['net_params'], inputs, mask)
            
            dv = jnp.concatenate([jnp.zeros((1,)), outputs], axis=0) # 2
            dv = dv*60.

            # dv = 1000*xn.mean(axis=0) # 2
            
            # dv = jnp.mean(vn, axis=0) - vi # 2
            # dv = jnp.array([0., 1.])*10

            dv = jnp.concatenate([dv, jnp.zeros(1)], axis=0) # 3
            dv = l2gr @ dv[:, None] # 3, 1
            dv = dv[:2, 0] # 2
            return dv

        dv = jax.vmap(get_dv)(x, v)

        v = v + dv * self.dt
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        x = x + self.speed * v * self.dt
        x = x%self.space_size # circular boundary
        return dict(x=x, v=v)

    def _step_state_simple(self, rng, state, params):
        x, v = state['x'], state['v'] # n_boids, 2

        # shape: src boids, tgt boids
        r = x[None, :, :] - x[:, None, :] # src, tgt, 2
        # r = jax.lax.select(r>0.5, r-1, jax.lax.select(r<-0.5, r+1, r))  # circular boundary
        r = jax.lax.select(r>self.space_size/2., r-self.space_size, jax.lax.select(r<-self.space_size/2, r+self.space_size, r))  # circular boundary
        d = jnp.linalg.norm(r, axis=-1) # src, tgt
        nbr_mask = (d<self.nbr_dist) * (1-jnp.eye(self.n_boids)) # src, tgt
        n_nbrs = nbr_mask.sum(axis=-1) # src
        
        # go towards neighbors' center
        nbr_center = (r * nbr_mask[..., None]).sum(axis=1) / n_nbrs[:, None] # src, 2
        acc_cohesion = jnp.where(n_nbrs[:, None]>0, nbr_center, jnp.zeros_like(v))

        nbr_mask_close = (d<self.nbr_dist_close) * (1-jnp.eye(self.n_boids)) # src, tgt
        n_nbrs_close = nbr_mask_close.sum(axis=-1) # src
        nbr_close = (-r * nbr_mask_close[..., None]).sum(axis=1) / n_nbrs_close[:, None] # src, 2
        acc_avoidance = jnp.where(n_nbrs_close[:, None]>0, nbr_close, jnp.zeros_like(v))

        # match neighbors avg velocity
        nbr_avg_v  = (v * nbr_mask[..., None]).sum(axis=1) / n_nbrs[:, None] # src, 2
        acc_alignment = jnp.where(n_nbrs[:, None]>0, (nbr_avg_v - v), jnp.zeros_like(v))

        acc = params['coef_cohesion'] * acc_cohesion + params['coef_avoidance'] * acc_avoidance + params['coef_alignment'] * acc_alignment
        v = v + acc * self.dt
        speed = jnp.linalg.norm(v, axis=-1, keepdims=True)
        # v = jnp.where(speed > self.max_speed, v/speed*self.max_speed, v)
        v = v/speed * self.max_speed
        x = x + v * self.dt
        x = x%1. # circular boundary
        return dict(x=x, v=v)
    
    def step_state(self, rng, state, params):
        if self.controller=='network':
            return self._step_state_network(rng, state, params)
        else:
            return self._step_state_simple(rng, state, params)
    
    def render_state(self, state, params, img_size=256):
        x, v = state['x'], state['v'] # n_boids, 2
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

        global2local, local2global = jax.vmap(get_transformation_mats)(x, v) # n_boids, 3, 3
        local_triangle_coords = jnp.array([[0, 1.], [0, -1.], [3, 0.]])*self.bird_render_size
        local_triangle_coords = jnp.concatenate([local_triangle_coords, jnp.ones((3, 1))], axis=-1)
        local_triangle_coords = local_triangle_coords[:, :, None] # 3, 3, 1

        global_triangle_coords = local2global[:, None, :, :] @ local_triangle_coords[None, :, :, :]
        global_triangle_coords = global_triangle_coords[:, :, :2, 0]
        img = jnp.ones((img_size, img_size, 3))

        x, y = jnp.linspace(0, self.space_size, img_size), jnp.linspace(0, self.space_size, img_size)
        x, y = jnp.meshgrid(x, y, indexing='ij')
        def render_triangle(img, triangle, color=[0., 0., 0.]):
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

            sharp = self.bird_render_sharpness
            mask = jax.nn.sigmoid(sharp*u) * jax.nn.sigmoid(sharp*v) * jax.nn.sigmoid(sharp*w)

            img = mask[..., None] * jnp.array(color) + (1-mask[..., None]) * img
            # mask = 1-mask.astype(jnp.float32)
            # img = jnp.minimum(img, mask[..., None])
            return img, None
        img, _ = jax.lax.scan(render_triangle, img, global_triangle_coords)
        if self.red_boid:
            img, _ = render_triangle(img, global_triangle_coords[0], color=[1., 0., 0.])
        return img


# class SimpleBoids():
#     def __init__(self, n_boids=64, max_speed=0.5, nbr_dist=0.1, nbr_dist_close=0.05,
#                  dt=0.01, bird_render_size=0.02, bird_render_sharpness=20):
#         self.n_boids = n_boids
#         self.max_speed = max_speed
#         self.nbr_dist = nbr_dist
#         self.nbr_dist_close = nbr_dist_close
#         self.dt = dt

#         self.bird_render_size = bird_render_size
#         self.bird_render_sharpness = bird_render_sharpness

#     def default_params(self, rng):
#         coef_cohesion = 0.005
#         coef_avoidance = 0.05
#         coef_alignment = 0.05
#         return dict(coef_cohesion=coef_cohesion, coef_avoidance=coef_avoidance, coef_alignment=coef_alignment)
        
#     def init_state(self, rng, params):
#         _rng1, _rng2, _rng3 = split(rng, 3)
#         x = jax.random.uniform(_rng1, (self.n_boids, 2), minval=0., maxval=1.)
#         v = jax.random.ball(_rng2, 2, 2, shape=(self.n_boids, ))
#         v = v/self.max_speed
#         return dict(x=x, v=v)
    
#     def step_state(self, rng, state, params):
#         x, v = state['x'], state['v'] # n_boids, 2

#         # shape: src boids, tgt boids
#         r = x[None, :, :] - x[:, None, :] # src, tgt, 2
#         r = jax.lax.select(r>0.5, r-1, jax.lax.select(r<-0.5, r+1, r))  # circular boundary
#         d = jnp.linalg.norm(r, axis=-1) # src, tgt
#         nbr_mask = (d<self.nbr_dist) * (1-jnp.eye(self.n_boids)) # src, tgt
#         n_nbrs = nbr_mask.sum(axis=-1) # src
        
#         # go towards neighbors' center
#         nbr_center = (r * nbr_mask[..., None]).sum(axis=1) / n_nbrs[:, None] # src, 2
#         acc_cohesion = jnp.where(n_nbrs[:, None]>0, nbr_center, jnp.zeros_like(v))

#         nbr_mask_close = (d<self.nbr_dist_close) * (1-jnp.eye(self.n_boids)) # src, tgt
#         n_nbrs_close = nbr_mask_close.sum(axis=-1) # src
#         nbr_close = (-r * nbr_mask_close[..., None]).sum(axis=1) / n_nbrs_close[:, None] # src, 2
#         acc_avoidance = jnp.where(n_nbrs_close[:, None]>0, nbr_close, jnp.zeros_like(v))

#         # match neighbors avg velocity
#         nbr_avg_v  = (v * nbr_mask[..., None]).sum(axis=1) / n_nbrs[:, None] # src, 2
#         acc_alignment = jnp.where(n_nbrs[:, None]>0, (nbr_avg_v - v), jnp.zeros_like(v))

#         acc = params['coef_cohesion'] * acc_cohesion + params['coef_avoidance'] * acc_avoidance + params['coef_alignment'] * acc_alignment
#         v = v + acc * self.dt
#         speed = jnp.linalg.norm(v, axis=-1, keepdims=True)
#         # v = jnp.where(speed > self.max_speed, v/speed*self.max_speed, v)
#         v = v/speed * self.max_speed
#         x = x + v * self.dt
#         x = x%1. # circular boundary
#         return dict(x=x, v=v)
    
#     def render_state(self, state, params, img_size=256):
#         x, v = state['x'], state['v'] # n_boids, 2
#         v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

#         global2local, local2global = jax.vmap(get_transformation_mats)(x, v) # n_boids, 3, 3
#         local_triangle_coords = jnp.array([[0, 1.], [0, -1.], [3, 0.]])*self.bird_render_size
#         local_triangle_coords = jnp.concatenate([local_triangle_coords, jnp.ones((3, 1))], axis=-1)
#         local_triangle_coords = local_triangle_coords[:, :, None] # 3, 3, 1

#         global_triangle_coords = local2global[:, None, :, :] @ local_triangle_coords[None, :, :, :]
#         global_triangle_coords = global_triangle_coords[:, :, :2, 0]
#         img = jnp.ones((img_size, img_size, 3))

#         x, y = jnp.linspace(0, 1, img_size), jnp.linspace(0, 1, img_size)
#         x, y = jnp.meshgrid(x, y, indexing='ij')
#         def render_triangle(img, triangle):
#             # Compute barycentric coordinates
#             v0 = triangle[2] - triangle[0]
#             v1 = triangle[1] - triangle[0]
#             v2 = jnp.stack([x, y], axis=-1) - triangle[0]
            
#             d00 = jnp.dot(v0, v0)
#             d01 = jnp.dot(v0, v1)
#             d11 = jnp.dot(v1, v1)
#             d20 = jnp.dot(v2, v0)
#             d21 = jnp.dot(v2, v1)
            
#             denom = d00 * d11 - d01 * d01
#             v = (d11 * d20 - d01 * d21) / denom
#             w = (d00 * d21 - d01 * d20) / denom
#             u = 1 - v - w
            
#             # Check if point is inside triangle
#             # mask = (u >= 0) & (v >= 0) & (w >= 0)

#             sharp = self.bird_render_sharpness
#             mask = jax.nn.sigmoid(sharp*u) * jax.nn.sigmoid(sharp*v) * jax.nn.sigmoid(sharp*w)

#             mask = 1-mask.astype(jnp.float32)
#             img = jnp.minimum(img, mask[..., None])
#             return img, None
#         img, _ = jax.lax.scan(render_triangle, img, global_triangle_coords)
#         return img
