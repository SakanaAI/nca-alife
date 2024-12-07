from functools import partial
import jax
import jax.numpy as jnp
from jax.random import split

from models.models_boids import Boids
from models.models_dnca import DNCA
from models.models_lenia import Lenia
from models.models_nca import NCA
from models.models_plenia import ParticleLenia
from models.models_plife import ParticleLife

import evosax

class FlattenSimulationParameters():
    def __init__(self, sim):
        self.sim = sim
        self.param_reshaper = evosax.ParameterReshaper(self.sim.default_params(jax.random.PRNGKey(0)))
        self.n_params = self.param_reshaper.total_params

    def default_params(self, rng):
        params = self.sim.default_params(rng)
        return self.param_reshaper.flatten_single(params)
    
    def init_state(self, rng, params):
        params = self.param_reshaper.reshape_single(params)
        return self.sim.init_state(rng, params)
    
    def step_state(self, rng, state, params):
        params = self.param_reshaper.reshape_single(params)
        return self.sim.step_state(rng, state, params)
    
    def render_state(self, state, params, img_size=None):
        params = self.param_reshaper.reshape_single(params)
        return self.sim.render_state(state, params, img_size)

def create_sim(sim_name):
    rollout_steps = 1000
    if sim_name=='boids':
        sim = Boids(n_boids=128, n_nbrs=16, visual_range=0.1, speed=0.5, controller='network', dt=0.01, bird_render_size=0.015, bird_render_sharpness=40.)
    elif sim_name.startswith('lenia'):
        _, clip = sim_name.split('_')
        sim = Lenia(grid_size=128, center_phenotype=True, phenotype_size=64, start_pattern="5N7KKM", clip1=float(clip))
        rollout_steps = 256
    elif sim_name=='plife_a':
        sim = ParticleLife(n_particles=5000, n_colors=6, search_space="alpha", dt=2e-3, render_radius=1e-2)  
    elif sim_name=='plife_ba':
        sim = ParticleLife(n_particles=5000, n_colors=6, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
    elif sim_name.startswith('plife_ba;'): # plife_ba;n=1000;k=1
        a, b, c = sim_name.split(';')
        n = int(b.split('=')[-1])
        k = int(c.split('=')[-1])
        sim = ParticleLife(n_particles=n, n_colors=k, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
    elif sim_name=='plife_ba_c3':
        sim = ParticleLife(n_particles=5000, n_colors=3, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
    elif sim_name=='dnca':
        sim = DNCA(grid_size=128, d_state=8, n_groups=1, identity_bias=0., temperature=1e-3)
    elif sim_name=='nca_d1':
        sim = NCA(grid_size=128, d_state=1, p_drop=0.5, dt=0.1)
    elif sim_name=='nca_d3':
        sim = NCA(grid_size=128, d_state=3, p_drop=0.5, dt=0.1)
    elif sim_name=='plenia':
        sim = ParticleLenia(n_particles=200, dt=0.1)
    else:
        raise ValueError(f"Unknown simulation name: {sim_name}")
    sim.sim_name = sim_name
    sim.rollout_steps = rollout_steps
    return sim

# def rollout_simulation(rng, params, sim, rollout_steps=None, img_size=128, ret='vid'):
#     if rollout_steps is None:
#         rollout_steps = sim.rollout_steps
#     def step(state, _rng):
#         next_state = sim.step_state(_rng, state, params)
#         return next_state, state
#     state_init = sim.init_state(rng, params)
#     state_final, state_vid = jax.lax.scan(step, state_init, split(rng, rollout_steps))
#     if ret=='vid':
#         vid = jax.vmap(partial(sim.render_state, params=params, img_size=img_size))(state_vid)
#         return vid
#     elif ret=='img':
#         img = sim.render_state(state_final, params=params, img_size=img_size)
#         return img

def rollout_simulation(rng, params, sim, rollout_steps, n_rollout_imgs='final', img_size=224,
                       return_state=False, chunk_ends=False, s0=None):
    if s0 is None:
        state_init = sim.init_state(rng, params)
    else:
        state_init = s0
    if n_rollout_imgs == 'final' or n_rollout_imgs=='image' or n_rollout_imgs == 'img':
        def step_fn(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, None
        state_final, _ = jax.lax.scan(step_fn, state_init, split(rng, rollout_steps))
        img = sim.render_state(state_final, params=params, img_size=img_size)
        if return_state:
            return dict(state_init=state_init, state_final=state_final, rgb=img)
        else:
            return dict(rgb=img)
    elif n_rollout_imgs == 'video' or n_rollout_imgs == 'vid':
        def step_fn(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, state
        state_final, state_vid = jax.lax.scan(step_fn, state_init, split(rng, rollout_steps))

        def render_state(_, state):
            return _, sim.render_state(state, params=params, img_size=img_size)
        _, vid = jax.lax.scan(render_state, None, state_vid)
        if return_state:
            return dict(state_init=state_init, state_final=state_final, state_vid=state_vid, rgb=vid)
        else:
            return dict(rgb=vid)
    else:
        def step_fn(state, _rng):
            next_state = sim.step_state(_rng, state, params)
            return next_state, state
        state_final, state_vid = jax.lax.scan(step_fn, state_init, split(rng, rollout_steps))
        chunk_size = rollout_steps//n_rollout_imgs
        if chunk_ends:
            idx_sample = jnp.arange(chunk_size-1, rollout_steps, chunk_size)
        else:
            idx_sample = jnp.arange(0, rollout_steps, chunk_size)
        state_vid = jax.tree.map(lambda x: x[idx_sample], state_vid)
        def render_state(_, state):
            return _, sim.render_state(state, params=params, img_size=img_size)
        _, vid = jax.lax.scan(render_state, None, state_vid)
        if return_state:
            return dict(state_init=state_init, state_final=state_final, state_vid=state_vid, rgb=vid)
        else:
            return dict(rgb=vid)

def rollout_and_embed_simulation(rng, params, sim, clip_model, rollout_steps, n_rollout_imgs='img',
                                  return_state=False, chunk_ends=False, s0=None):
    data = rollout_simulation(rng, params, sim, rollout_steps, n_rollout_imgs, img_size=224,
                              return_state=return_state, chunk_ends=chunk_ends, s0=s0)
    if clip_model is None:
        return dict(**data, z=None)
    elif n_rollout_imgs == 'final':
        z = clip_model.embed_img(data['rgb'])
        return dict(**data, z=z)
    else:
        z = jax.vmap(clip_model.embed_img)(data['rgb'])
        return dict(**data, z=z)


if __name__ == '__main__':
    from tqdm.auto import tqdm
    names = ['boids', 'dnca', 'lenia', 'nca_d1', 'nca_d3', 'plenia', 'plife_a', 'plife_ba', 'plife_ba_c3']

    for name in names:
        sim = create_sim(name)
        print(name, sim.rollout_steps)

        rng = jax.random.PRNGKey(0)
        state = sim.init_state(rng, sim.default_params(rng))

        def step(state, _rng):
            next_state = sim.step_state(_rng, state, sim.default_params(rng))
            return next_state, state
        
        jax.jit(step)(state, rng)

        print(name)
        for _ in tqdm(range(100)):
            state, _ = jax.lax.scan(step, state, jax.random.split(rng, sim.rollout_steps))
        print()

