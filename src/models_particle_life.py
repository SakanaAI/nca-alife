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
        # NOTE: the particle space is [0, 1]
    
    def get_random_env_params(self, rng, distribution='uniform'):
        beta = jnp.array(0.3)
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
                         color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='k',
                         **kwargs):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = mcolors.to_rgb(background_color)
        
        pos, vel, pcol = state['x'], state['v'], state['c']
        color = color_palette[pcol]

        plt.scatter(*(pos*img_size).T, c=color, s=radius, **kwargs)
        # plt.xlim(-0.0, img_size)
        # plt.ylim(-0.0, img_size)

    def render_state_light(self, state, img_size=256, radius=0,
                           color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='k'):
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
        
    def render_state_heavy(self, state, img_size=256, radius=0.5, sharpness=10.,
                           color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='k'):
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
            coeff = 1.- (1./(1.+jnp.exp(-sharpness*(d-radius))))
            img = coeff[:, :, None]*color + (1-coeff[:, :, None])*img
            return img, None
    
        x, y = pos.T * img_size
        radius = jnp.full(x.shape, fill_value=radius)
        color = color_palette[pcol]
        img, _ = jax.lax.scan(render_circle, img, (x, y, radius, color))
        return img



####### testing out Jax-MD:
# def main_jaxmd(args):
#     from jax_md import partition
#     from jax_md import space
    
#     box_size = 1.
#     cell_size = 0.1
#     displacement_fn, shift_fn = space.periodic(box_size)
#     neighbor_list_fn = partition.neighbor_list(displacement_fn, box_size, cell_size, capacity_multiplier=12.5)

#     rng = jax.random.PRNGKey(0)
#     R = jax.random.uniform(rng, (1000, 2), minval=0., maxval=1.)
#     neighbors = neighbor_list_fn.allocate(R) # Create a new neighbor list.
#     print(neighbors.idx.shape)

#     rng, _rng = split(rng)
#     R = jax.random.uniform(rng, (1000, 2), minval=0., maxval=1.)
#     neighbors = neighbors.update(R)
#     print(neighbors.idx.shape)
#     print(neighbors.idx)

#     print('here')
#     for i in range(100):
#         Rb = R[neighbors.idx[:, i]]
#         d = jnp.linalg.norm(R-Rb, axis=-1)
#         print(d.mean())



    
if __name__=='__main__':
    import numpy as np
    from tqdm.auto import tqdm
    plife = ParticleLife(6000, 6, n_dims=2, dt=0.001, half_life=0.04, rmax=0.1)

    rng = jax.random.PRNGKey(0)
    env_params = plife.get_random_env_params(rng)

    # print(env_params['alphas'])
    # env_params['alphas'] = jnp.array([[0, -1., 0], [-1, 0, 0], [0.5, 0.5, 1.]])
    # env_params['alphas'] = jnp.array([[.5, .25, -.2], [1., .5, .1], [-0.5, 0.5, .25]])
    # print(env_params['alphas'])

    print('env_params', jax.tree.map(lambda x: x.shape, env_params))

    state = plife.get_random_init_state(rng)
    def step(state, _):
        state = plife.forward_step(state, env_params)
        return state, state

    state, statevid = jax.lax.scan(step, state, length=1000)
    print('statevid', jax.tree.map(lambda x: x.shape, statevid))
    # statevid = jax.tree.map(lambda x: x[::10], statevid)
    # print('statevid', jax.tree.map(lambda x: x.shape, statevid))

    color_palette = 'FF0000-00FF00-0000FF-FFFF00-00FFFF-FF00FF-800000-808000'
    # render_fn = partial(plife.render_state_heavy, img_size=1024, radius=2., sharpness=10., color_palette=color_palette)
    render_fn = partial(plife.render_state_heavy, img_size=512, radius=1., sharpness=10., color_palette=color_palette)
    vid = jax.vmap(render_fn)(statevid)

    print('vid', jax.tree.map(lambda x: x.shape, vid))
    vid = np.array((vid*255).astype(jnp.uint8))
    print('np vid', jax.tree.map(lambda x: x.shape, vid))
    import imageio
    imageio.mimwrite(f'./temp/vid.mp4', vid, fps=100, codec='libx264')

