
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split

from collections import namedtuple
import jax.numpy as jp

from functools import partial

def inv_sigmoid(x):
    return jnp.log(x) - jnp.log1p(-x)

Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')

def peak_f(x, mu, sigma):
  return jp.exp(-((x-mu)/sigma)**2)

def fields_f(p: Params, points, x):
  r = jp.sqrt(jp.square(x-points).sum(-1).clip(1e-10))
  U = peak_f(r, p.mu_k, p.sigma_k).sum()*p.w_k
  G = peak_f(U, p.mu_g, p.sigma_g)
  R = p.c_rep/2 * ((1.0-r).clip(0.0)**2).sum()
  return Fields(U, G, R, E=R-G)

def motion_f(params, points):
  grad_E = jax.grad(lambda x : fields_f(params, points, x).E)
  return -jax.vmap(grad_E)(points)

def odeint_euler(f, params, x0, dt, n):
  def step_f(x, _):
    x = x+dt*f(params, x)
    return x, x
  return jax.lax.scan(step_f, x0, None, n)[1]

def lerp(x, a, b):
  return jp.float32(a)*(1.0-x) + jp.float32(b)*x
def cmap_e(e):
  return 1.0-jp.stack([e, -e], -1).clip(0) @ jp.float32([[0.3,1,1], [1,0.3,1]])
def cmap_ug(u, g):
  vis = lerp(u[...,None], [0.1,0.1,0.3], [0.2,0.7,1.0])
  return lerp(g[...,None], vis, [1.17,0.91,0.13])

def vmap2(f):
  return jax.vmap(jax.vmap(f))

@partial(jax.jit, static_argnames=['w', 'show_UG', 'show_cmap'])
def show_lenia(params, points, extent, w=400, show_UG=False, show_cmap=True):
  xy = jp.mgrid[-1:1:w*1j, -1:1:w*1j].T*extent
  e0 = -peak_f(0.0, params.mu_g, params.sigma_g)
  f = partial(fields_f, params, points)
  fields = vmap2(f)(xy)
  r2 = jp.square(xy[...,None,:]-points).sum(-1).min(-1)
  points_mask = (r2/0.02).clip(0, 1.0)[...,None]
  vis = cmap_e(fields.E-e0) * points_mask
  if show_cmap:
    e_mean = jax.vmap(f)(points).E.mean()
    bar = np.r_[0.5:-0.5:w*1j]
    bar = cmap_e(bar) * (1.0-peak_f(bar, e_mean-e0, 0.005)[:,None])
    vis = jp.hstack([vis, bar[:,None].repeat(16, 1)])
  if show_UG:
    vis_u = cmap_ug(fields.U, fields.G)*points_mask
    if show_cmap:
      u = np.r_[1:0:w*1j]
      bar = cmap_ug(u, peak_f(u, params.mu_g, params.sigma_g))
      bar = bar[:,None].repeat(16, 1)
      vis_u = jp.hstack([bar, vis_u])
    vis = jp.hstack([vis_u, vis])
  return vis


class ParticleLenia():
    def __init__(self, n_particles=256,):
        self.n_particles = n_particles
        self.dt = 0.1

    def default_params(self, rng):
        params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
        return params
    
    def init_state(self, rng, params):
        state = (jax.random.uniform(rng, [self.n_particles, 2])-0.5)*12.0
        return state
    
    def step_state(self, rng, state, params):
        state = state + self.dt * motion_f(params, state)
        return state
    
    def render_state(self, state, params, img_size=None):
        extent = jp.abs(state).max()*1.2
        img = show_lenia(params, state, extent=extent, w=img_size, show_UG=True, show_cmap=False)
        img = img[:, :img_size, :]
        
        a, b = img.min(), img.max()
        img = (img-a)/(b-a)
        # if img_size is not None:
            # img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img



