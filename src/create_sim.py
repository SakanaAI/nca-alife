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

def create_sim(sim_name):
    rollout_steps = 1000
    if sim_name=='boids':
        sim = Boids(n_boids=128, n_nbrs=16, visual_range=0.1, speed=0.5, controller='network', dt=0.01, bird_render_size=0.015, bird_render_sharpness=40.)
    elif sim_name=='dnca':
        sim = DNCA(grid_size=128, d_state=8, n_groups=1, identity_bias=0., temperature=1e-3)
    elif sim_name.startswith('lenia'):
        _, clip = sim_name.split('_')
        sim = Lenia(grid_size=128, center_phenotype=True, phenotype_size=64, start_pattern="5N7KKM", clip1=float(clip))
        rollout_steps = 256
    elif sim_name=='nca_d1':
        sim = NCA(grid_size=128, d_state=1, p_drop=0.5, dt=0.1)
    elif sim_name=='nca_d3':
        sim = NCA(grid_size=128, d_state=3, p_drop=0.5, dt=0.1)
    elif sim_name=='plenia':
        sim = ParticleLenia(n_particles=200, dt=0.1)
    elif sim_name=='plife_a':
        sim = ParticleLife(n_particles=5000, n_colors=6, search_space="alpha", dt=2e-3, render_radius=1e-2)  
    elif sim_name=='plife_ba':
        sim = ParticleLife(n_particles=5000, n_colors=6, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
    elif sim_name=='plife_ba_c3':
        sim = ParticleLife(n_particles=5000, n_colors=3, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
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

def rollout_render_simulation(rng, params, sim, rollout_steps, n_rollout_imgs='img', img_size=224):
    def step(state, _rng):
        next_state = sim.step_state(_rng, state, params)
        return next_state, state
    state_init = sim.init_state(rng, params)
    state_final, state_vid = jax.lax.scan(step, state_init, split(rng, rollout_steps))
    if n_rollout_imgs == 'img':
        return sim.render_state(state_final, params=params, img_size=img_size)
    elif n_rollout_imgs == 'vid':
        return jax.vmap(partial(sim.render_state, params=params, img_size=img_size))(state_vid)
    else:
        sr = rollout_steps//n_rollout_imgs # sampling rate
        idx_sample = jnp.arange(sr-1, rollout_steps, sr)
        state_vid = jax.tree.map(lambda x: x[idx_sample], state_vid)
        vid = jax.vmap(partial(sim.render_state, params=params, img_size=img_size))(state_vid)
        return vid

def rollout_render_clip_simulation(rng, params, sim, clip_model, rollout_steps, n_rollout_imgs=None):
    vid = rollout_render_simulation(rng, params, sim, rollout_steps, n_rollout_imgs, img_size=224)
    if clip_model is None:
        return dict(vid=vid, z=None)

    if n_rollout_imgs is None:
        z = clip_model.embed_img(vid)
    else:
        z = jax.vmap(clip_model.embed_img)(vid)
    return dict(vid=vid, z=z)

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

