import jax

from models.models_boids import Boids
from models.models_dnca import DNCA
from models.models_lenia import Lenia
from models.models_nca import NCA
from models.models_plenia import ParticleLenia
from models.models_plife import ParticleLife

def create_sim(name):
    if name=='boids':
        sim = Boids(n_boids=128, n_nbrs=16, visual_range=0.1, speed=0.5, controller='network', dt=0.01, bird_render_size=0.015, bird_render_sharpness=40.)
        sim.rollout_steps = 1000
    elif name=='dnca':
        sim = DNCA(grid_size=128, d_state=8, n_groups=1, identity_bias=0., temperature=1e-3)
        sim.rollout_steps = 1000
    elif name=='lenia':
        sim = Lenia(grid_size=128, center_phenotype=True, phenotype_size=64, start_pattern="5N7KKM")
        sim.rollout_steps = 256
    elif name=='nca_d1':
        sim = NCA(grid_size=128, d_state=1, p_drop=0.5, dt=0.1)
        sim.rollout_steps = 1000
    elif name=='nca_d3':
        sim = NCA(grid_size=128, d_state=3, p_drop=0.5, dt=0.1)
        sim.rollout_steps = 1000
    elif name=='plenia':
        sim = ParticleLenia(n_particles=200, dt=0.1)
        sim.rollout_steps = 1000
    elif name=='plife_a':
        sim = ParticleLife(n_particles=5000, n_colors=6, search_space="alpha", dt=2e-3, render_radius=1e-2)  
        sim.rollout_steps = 1000
    elif name=='plife_ba':
        sim = ParticleLife(n_particles=5000, n_colors=6, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
        sim.rollout_steps = 1000
    elif name=='plife_ba_c3':
        sim = ParticleLife(n_particles=5000, n_colors=3, search_space="beta+alpha", dt=2e-3, render_radius=1e-2)  
        sim.rollout_steps = 1000
    else:
        raise ValueError(f"Unknown simulation name: {name}")
    return sim


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

