import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat

from functools import partial


class ParticleLife():
    def __init__(self, n_particles, n_colors, n_dims=2, dt=0.01, half_life=0.04, rmax=0.1):
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_dims = n_dims
        self.dt = dt
        self.half_life = half_life
        self.rmax = rmax
        # NOTE: the actual space is [0, 1]
    
    def get_random_env_params(self, rng, distribution='uniform'):
        beta = 0.3
        if distribution=='uniform':
            alphas = jax.random.uniform(rng, (self.n_colors, self.n_colors), minval=-1., maxval=1.)
        elif distribution=='normal':
            alphas = jax.random.normal(rng, (self.n_colors, self.n_colors))
        masses = jnp.ones((self.n_colors, ))/10.
        return dict(alphas=alphas, beta=beta, masses=masses)
    
    def get_random_init_state(self, rng, distribution='uniform'):
        rng, _rng = split(rng)
        c = jax.random.randint(_rng, (self.n_particles,), 0, self.n_colors)
        rng, _rng = split(rng)
        if distribution=='uniform':
            x = jax.random.uniform(_rng, (self.n_particles, self.n_dims), minval=0, maxval=1)
        v = jnp.zeros((self.n_particles, self.n_dims))
        return dict(c=c, x=x, v=v)

    def _calc_force(self, r, alpha, beta):
        first = r / beta - 1
        second = alpha * (1 - jnp.abs(2 * r - 1 - beta) / (1 - beta))
        cond_first = (r < beta) # (0 <= r) & (r < beta)
        cond_second = (beta < r) & (r < 1)
        return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))
    
    def forward_step(self, state, env_params):
        x, v, c = state['x'], state['v'], state['c']
        alphas, beta, masses = env_params['alphas'], env_params['beta'], env_params['masses']
        
        mass = masses[c]
        # index alphas using all pairwise colors, code looks weird, but works
        alpha = alphas[c[:, None], c[None, :]] # (N, N)
        r = x[None, :, :] - x[:, None, :] # (N, N, D)
        r = jax.lax.select(r>0.5, r-1, jax.lax.select(r<-0.5, r+1, r))
        # print(r[2, 4], (x[4] - x[2])); print(alpha[2,4], alphas[c[2], c[4]])
        rlen = jnp.linalg.norm(r, axis=-1)
    
        # double vmap for pairwise (N, N) interactions
        f = jax.vmap(jax.vmap(partial(self._calc_force, beta=beta)))(rlen/self.rmax, alpha)
        f = r/(rlen[..., None]+1e-8) * f[..., None]
        # f = f.at[jnp.arange(n_particles), jnp.arange(n_particles)].set(0.) # uneeded if adding epsilon term
        
        f = self.rmax * jnp.sum(f, axis=1)
        acc = f / mass[:, None]
        
        mu = (0.5) ** (self.dt / self.half_life)
        v = mu * v + acc * self.dt
        x = x + v * self.dt
    
        x = x%1. # circular boundary
        return dict(c=c, x=x, v=v)
    
    def render_state_mpl(self, state, img_size=256, radius=10.,
                         color_palette='264653-287271-2a9d8f-8ab17d-e9c46a-f4a261-ee8959-e76f51', background_color='k',
                         **kwargs):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = mcolors.to_rgb(background_color)
        
        pos, vel, pcol = state['x'], state['v'], state['c']
        color = color_palette[pcol]

        plt.scatter(*(pos*img_size).T, c=color, s=radius, **kwargs)
        # plt.xlim(-0.0, img_size)
        # plt.ylim(-0.0, img_size)

    def render_state_light(self, state, img_size=256, radius=0,
                           color_palette='264653-287271-2a9d8f-8ab17d-e9c46a-f4a261-ee8959-e76f51', background_color='k'):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = jnp.array(mcolors.to_rgb(background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)
        
        pos, vel, pcol = state['x'], state['v'], state['c']
        xmid, ymid = (pos.T*img_size).astype(int)
        x, y = [], []
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                x.append(xmid+i)
                y.append(ymid+j)
        x, y = jnp.concatenate(x), jnp.concatenate(y)
        color = color_palette[pcol]
        color = repeat(color, "N D -> (K N) D", K=(radius*2+1)**2)
        img = img.at[x, y, :].set(color)
        return img
        
    def render_state_heavy(self, state, img_size=256, radius=3, blur=1.,
                           color_palette='264653-287271-2a9d8f-8ab17d-e9c46a-f4a261-ee8959-e76f51', background_color='k'):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = jnp.array(mcolors.to_rgb(background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)
        
        pos, vel, pcol = state['x'], state['v'], state['c']
        xgrid = ygrid = jnp.arange(img_size)
        xgrid, ygrid = jnp.meshgrid(xgrid, ygrid, indexing='ij')
    
        def render_circle(img, circle_data):
            x, y, radius, color = circle_data
            d2 = (x-xgrid)**2 + (y-ygrid)**2
            d = jnp.sqrt(d2)
            
            # d2 = (d2<radius**2).astype(jnp.float32)[:, :, None]
            # img = d2*color + (1.-d2)*img
            coeff = 1.- (1./(1.+jnp.exp(-blur*(d-radius))))
            img = coeff[:, :, None]*color + (1-coeff[:, :, None])*img
            return img, None
    
        x, y = pos.T * img_size
        radius = jnp.full(x.shape, fill_value=radius)
        color = color_palette[pcol]
        img, _ = jax.lax.scan(render_circle, img, (x, y, radius, color))
        return img
    
    
    
