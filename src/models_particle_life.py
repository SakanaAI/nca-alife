import jax
import jax.numpy as jnp
from jax.random import split

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from einops import repeat

from functools import partial

class ParticleLife():
    def __init__(self, n_particles, n_colors, n_dims=2):
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_dims = n_dims
        # NOTE: the particle space is [0, 1]
    
    def get_default_env_params(self):
        return dict(
            beta=jnp.full((self.n_colors, ), 0.3),
            alpha=jnp.zeros((self.n_colors, self.n_colors)),
            mass=jnp.full((self.n_colors, ), 0.1),
            dt=jnp.array(0.01),
            half_life=jnp.full((self.n_colors, ), 0.04),
            rmax=jnp.full((self.n_colors, ), 0.1),
        )
    # def get_random_env_params(self, rng, distribution='uniform'):
    #     beta = jnp.array(0.3)
    #     if distribution=='uniform':
    #         alphas = jax.random.uniform(rng, (self.n_colors, self.n_colors), minval=-1., maxval=1.)
    #     elif distribution=='normal':
    #         alphas = jax.random.normal(rng, (self.n_colors, self.n_colors))
    #     masses = jnp.ones((self.n_colors, ))/10.
    #     return dict(alphas=alphas, beta=beta, masses=masses)
    
    def get_random_init_state(self, rng, distribution='uniform'):
        rng, _rng = split(rng)
        c = jax.random.randint(_rng, (self.n_particles,), 0, self.n_colors)
        rng, _rng = split(rng)
        if distribution=='uniform':
            x = jax.random.uniform(_rng, (self.n_particles, self.n_dims), minval=0, maxval=1)
        v = jnp.zeros((self.n_particles, self.n_dims))
        return dict(c=c, x=x, v=v)
    
    def forward_step(self, state, env_params):
        x, v, c = state['x'], state['v'], state['c']

        mass = env_params['mass'][c]
        half_life = env_params['half_life'][c]
        dt = env_params['dt']

        def force_graph(r, alpha, beta):
            first = r / beta - 1
            second = alpha * (1 - jnp.abs(2 * r - 1 - beta) / (1 - beta))
            cond_first = (r < beta) # (0 <= r) & (r < beta)
            cond_second = (beta < r) & (r < 1)
            return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))
        def calc_force(x1, x2, c1, c2):
            r = x2 - x1
            r = jax.lax.select(r>0.5, r-1, jax.lax.select(r<-0.5, r+1, r))  # circular boundary

            alpha, beta, rmax = env_params['alpha'][c1, c2], env_params['beta'][c1], env_params['rmax'][c1]
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
        
    def render_state_heavy(self, state, env_params, img_size=256, radius=4e-3, sharpness=1.5e3,
                           color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='k'):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = jnp.array(mcolors.to_rgb(background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)
        
        pos, vel, pcol = state['x'], state['v'], state['c']
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
    
        x, y = pos.T
        radius = jnp.sqrt(env_params['mass'][pcol]) * radius
        color = color_palette[pcol]
        img, _ = jax.lax.scan(render_circle, img, (x, y, radius, color))
        return img


if __name__=='__main__f':
    import numpy as np
    from tqdm.auto import tqdm
    from einops import rearrange

    plife = ParticleLife(10000, 6, n_dims=2, dt=0.001, half_life=0.04, rmax=0.1)

    rng = jax.random.PRNGKey(0)
    env_params = plife.get_random_env_params(rng)

    state = plife.get_random_init_state(rng)
    def step(state, _):
        state = plife.forward_step(state, env_params)
        return state, None
    state, _ = jax.lax.scan(step, state, length=10000)


    def custom_render(state, img_size=512, window_size=11,
                      color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='k'):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = jnp.array(mcolors.to_rgb(background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)

        pos, vel, pcol = state['x'], state['v'], state['c']

        pos = pos*img_size

        ws = window_size//2

        a = jnp.arange(-ws, ws+1)
        kx, ky = jnp.meshgrid(a, a, indexing='ij')
        kxy = jnp.stack([kx, ky], axis=-1)

        sharpness = 10.
        radius = 4.

        def get_point_window_and_loc(x, c):
            window = repeat(background_color, "C -> H W C", H=window_size, W=window_size)
            loc = x.astype(int)
            my_kxy = kxy + loc
            d = jnp.linalg.norm(my_kxy - x, axis=-1)
            coeff = 1./(1.+jnp.exp(sharpness * (d - radius)))
            window = coeff[:, :, None] * c + (1-coeff[:, :, None]) * window
            return window, my_kxy

        window, k_xy = jax.vmap(get_point_window_and_loc)(pos, color_palette[pcol])
        window = rearrange(window, "N H W C -> (N H W) C")
        k_xy = rearrange(k_xy, "N H W two -> two (N H W)")
        x, y = k_xy
        img = img.at[x, y, :].set(window)
        return img
    
    def custom_render2(state, img_size=512, window_size=11,
                      color_palette='ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00', background_color='k'):
        color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in color_palette.split('-')])
        background_color = jnp.array(mcolors.to_rgb(background_color)).astype(jnp.float32)
        img = repeat(background_color, "C -> H W C", H=img_size, W=img_size)

        pos, vel, pcol = state['x'], state['v'], state['c']

        pos = pos*img_size

        ws = window_size//2

        a = jnp.arange(-ws, ws+1)
        kx, ky = jnp.meshgrid(a, a, indexing='ij')
        kxy = jnp.stack([kx, ky], axis=-1)

        sharpness = 10.
        radius = 4.

        def get_point_window_and_loc(x, c):
            window = repeat(background_color, "C -> H W C", H=window_size, W=window_size)
            loc = x.astype(int)
            my_kxy = kxy + loc
            d = jnp.linalg.norm(my_kxy - x, axis=-1)
            coeff = 1./(1.+jnp.exp(sharpness * (d - radius)))
            window = coeff[:, :, None] * c + (1-coeff[:, :, None]) * window
            return window, my_kxy

        window, k_xy = jax.vmap(get_point_window_and_loc)(pos, color_palette[pcol])
        window = rearrange(window, "N H W C -> N (H W) C")
        x, y = rearrange(k_xy, "N H W two -> two N (H W)")

        def render_img(img, data):
            window, x, y = data

            window = jnp.maximum(img[x, y, :], window)
            img = img.at[x, y, :].set(window)
            return img, None
        
        img, _ = jax.lax.scan(render_img, img, (window, x, y))
        return img
        

    render_fns = [
        partial(plife.render_state_light, img_size=512, radius=0),
        partial(plife.render_state_heavy, img_size=512, radius=1., sharpness=10.),
        custom_render,
        custom_render2
    ]
    render_fns = [jax.jit(render_fn) for render_fn in render_fns]
 
    for render_fn in render_fns:
        for i in tqdm(range(100)):
            render_fn(state)

    imgs = np.stack([render_fn(state) for render_fn in render_fns])
    imgs[:, :, -1, :] = [1, 0, 0]
    imgs = rearrange(imgs, 'N H W C -> H (N W) C')
    plt.imsave('./temp/plife.png', imgs)


if __name__=='__main__':
    kernel_designs = dict(
        phases=jnp.array([[0, -1., 0], [-1, 0, 0], [0.5, 0.5, 1.]]),
        gliders=jnp.array([[.5, .25, -.2], [1., .5, .1], [-0.5, 0.5, .25]]),
        caterpillar=jnp.array([
            [1, .2, 0, 0, 0, 0],
            [0, 1, .2, 0, 0, 0],
            [0, 0, 1, .2, 0, 0],
            [0, 0, 0, 1, .2, 0],
            [0, 0, 0, 0, 1, .2],
            [.2, 0, 0, 0, 0, 1]
        ])
    )

    import numpy as np
    from tqdm.auto import tqdm

    for name, kernel in kernel_designs.items():
        print(name)
        plife = ParticleLife(6000, len(kernel))

        rng = jax.random.PRNGKey(0)
        env_params = plife.get_default_env_params()
        env_params['alpha'] = kernel
        env_params['dt'] = 0.003

        state = plife.get_random_init_state(rng)
        def step(state, _):
            state = plife.forward_step(state, env_params)
            return state, state

        state, statevid = jax.lax.scan(step, state, length=2000)

        render_fn = partial(plife.render_state_heavy, img_size=256)
        vid = jax.vmap(render_fn, in_axes=(0, None))(statevid, env_params)
        print('vid', jax.tree.map(lambda x: x.shape, vid))
        print(vid.max(), vid.min(), vid.dtype)
        vid = np.array((vid*255).astype(jnp.uint8))
        import imageio
        imageio.mimwrite(f'./temp/vid_{name}.mp4', vid, fps=100, codec='libx264')


