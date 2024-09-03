import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat, rearrange

from functools import partial

# NOTE: the particle space is [0, 1]
class ParticleLife():
    def __init__(self, n_particles=5000, n_colors=6, n_dims=2, x_dist_bins=7,
                 beta=0.3, alpha=0., mass=0.1,
                 dt=0.002, half_life=0.04, rmax=0.1,
                 render_radius=1e-2, sharpness=20.,
                 search_space="beta+alpha+mass+dt+half_life+rmax+c_dist+x_dist+color+render_radius",
                 color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='black'):
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_dims = n_dims
        assert n_dims==2, 'only 2d supported for now'
        self.x_dist_bins = x_dist_bins

        self.search_space = search_space.split('+')
        self.color_palette = color_palette
        self.background_color = background_color

        self.render_radius = render_radius
        self.sharpness = sharpness

        self.fixed_params = dict(
            beta=jnp.full((self.n_colors, ), beta),
            alpha=jnp.full((self.n_colors, self.n_colors), alpha),
            mass=jnp.full((self.n_colors, ), mass),
            dt=jnp.array(dt),
            half_life=jnp.full((self.n_colors, ), half_life),
            rmax=jnp.full((self.n_colors, ), rmax),
            c_dist=jnp.zeros((self.n_colors, )),
            x_dist=jnp.zeros((self.n_colors, self.x_dist_bins, self.x_dist_bins)),
        )

    def _get_param(self, params, name):
        if name in self.search_space:
            if name=="beta":
                return jax.nn.sigmoid(params[name]) # 0. to 1.
            elif name=="alpha":
                return jax.nn.sigmoid(params[name]+0.)*3.-1.5
            elif name=="mass":
                return 10.**(jax.nn.sigmoid(params[name])*1.5-1.5) # .03 to 1
            elif name=="dt":
                return 10.**(jax.nn.sigmoid(params[name])*3.-4.) # 1e-4 to 1e-1
            elif name=="half_life":
                return 10.**(jax.nn.sigmoid(params[name])*1.5-2.5) # .003 to .1
            elif name=="rmax":
                return 10.**(jax.nn.sigmoid(params[name])*1.5-2.) # .01 to 3e-1
            elif name=="c_dist":
                return params[name]
            elif name=="x_dist":
                return params[name]
        else:
            return self.fixed_params[name]
    
    # def default_params(self, rng):
    #     alpha = jax.random.uniform(rng, (self.n_colors, self.n_colors), minval=-1., maxval=1.)
    #     return dict(
    #         beta=jnp.full((self.n_colors, ), 0.3),
    #         alpha=alpha, #jnp.zeros((self.n_colors, self.n_colors)),
    #         mass=jnp.full((self.n_colors, ), 0.1),
    #         dt=jnp.array(0.002),
    #         half_life=jnp.full((self.n_colors, ), 0.04),
    #         rmax=jnp.full((self.n_colors, ), 0.1),
    #         c_dist=jnp.zeros((self.n_colors, )),
    #         x_dist=jnp.zeros((self.n_colors, self.x_dist_bins, self.x_dist_bins)),
    #     )
    def default_params(self, rng):
        params = {}
        if "beta" in self.search_space:
            rng, _rng = split(rng)
            params['beta'] = jax.random.normal(_rng, (self.n_colors, ))
        if "alpha" in self.search_space:
            rng, _rng = split(rng)
            params['alpha'] = jax.random.normal(_rng, (self.n_colors, self.n_colors))
        if "mass" in self.search_space:
            rng, _rng = split(rng)
            params['mass'] = jax.random.normal(_rng, (self.n_colors, ))
        if "dt" in self.search_space:
            rng, _rng = split(rng)
            params['dt'] = jax.random.normal(_rng, ())
        if "half_life" in self.search_space:
            rng, _rng = split(rng)
            params['half_life'] = jax.random.normal(_rng, (self.n_colors, ))
        if "rmax" in self.search_space:
            rng, _rng = split(rng)
            params['rmax'] = jax.random.normal(_rng, (self.n_colors, ))
        if "c_dist" in self.search_space:
            rng, _rng = split(rng)
            params['c_dist'] = jax.random.normal(_rng, (self.n_colors, ))
        if "x_dist" in self.search_space:
            rng, _rng = split(rng)
            params['x_dist'] = jax.random.normal(_rng, (self.n_colors, self.x_dist_bins, self.x_dist_bins))
        return params
        
    def init_state(self, rng, params):
        _rng1, _rng2, _rng3 = split(rng, 3)
        c_dist = jax.nn.softmax(self._get_param(params, 'c_dist'))
        c = jax.random.choice(_rng1, self.n_colors, shape=(self.n_particles, ), replace=True, p=c_dist)

        x = jax.random.uniform(_rng2, (self.n_particles, self.n_dims), minval=0., maxval=1./self.x_dist_bins)
        x_dist = rearrange(self._get_param(params, 'x_dist')[c], "N H W -> N (H W)")
        x_bin = jax.random.categorical(_rng3, x_dist, axis=-1)
        xbin, ybin = jnp.unravel_index(x_bin, (self.x_dist_bins, self.x_dist_bins))
        xbin = jnp.stack([xbin, ybin], axis=-1)
        x = x + xbin * (1./self.x_dist_bins)

        v = jnp.zeros((self.n_particles, self.n_dims))
        return dict(c=c, x=x, v=v)
    
    def step_state(self, rng, state, params):
        x, v, c = state['x'], state['v'], state['c']

        mass = self._get_param(params, 'mass')[c]
        half_life = self._get_param(params, 'half_life')[c]
        dt = self._get_param(params, 'dt')

        param_alpha = self._get_param(params, 'alpha')
        param_beta = self._get_param(params, 'beta')
        param_rmax = self._get_param(params, 'rmax')

        def force_graph(r, alpha, beta):
            first = r / beta - 1
            second = alpha * (1 - jnp.abs(2 * r - 1 - beta) / (1 - beta))
            cond_first = (r < beta) # (0 <= r) & (r < beta)
            cond_second = (beta < r) & (r < 1)
            return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))
        def calc_force(x1, x2, c1, c2):
            r = x2 - x1
            r = jax.lax.select(r>0.5, r-1, jax.lax.select(r<-0.5, r+1, r))  # circular boundary

            alpha, beta, rmax = param_alpha[c1, c2], param_beta[c1], param_rmax[c1]
            rlen = jnp.linalg.norm(r)
            rdir = r / (rlen + 1e-8)
            flen = rmax * force_graph(rlen/rmax, alpha, beta)
            return rdir * flen
        
        f = jax.vmap(jax.vmap(calc_force, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))(x, x, c, c)
        acc = f.sum(axis=-2) / mass[:, None]
        
        mu = (0.5) ** (dt / half_life[:, None])
        v = mu * v + acc * dt
        x = x + v * dt
        x = x%1. # circular boundary
        return dict(c=c, x=x, v=v)
    
    def render_state(self, state, params, img_size=256):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in self.color_palette.split('-')])
        background_color = jnp.array(mcolors.to_rgb(self.background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)

        render_radius = self.render_radius
        sharpness = self.sharpness / render_radius

        x, v, c = state['x'], state['v'], state['c']
        mass = self._get_param(params, 'mass')[c]
        i = jnp.argsort(mass)[::-1]

        x, v, c, mass = x[i], v[i], c[i], mass[i]
        color = color_palette[c]

        xgrid = ygrid = jnp.linspace(0, 1, img_size)
        xgrid, ygrid = jnp.meshgrid(xgrid, ygrid, indexing='ij')

        def render_circle(img, circle_data):
            x, y, radius, color = circle_data
            d2 = (x-xgrid)**2 + (y-ygrid)**2
            d = jnp.sqrt(d2)
            # d2 = (d2<radius**2).astype(jnp.float32)[:, :, None]
            # img = d2*color + (1.-d2)*img
            coeff = 1.- (1./(1.+jnp.exp(-sharpness*(d-radius))))
            img = coeff[:, :, None]*color + (1-coeff[:, :, None])*img
            return img, None
    
        radius = jnp.sqrt(mass) * render_radius
        img, _ = jax.lax.scan(render_circle, img, (*x.T, radius, color))
        return img
    

