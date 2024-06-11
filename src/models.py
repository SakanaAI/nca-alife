import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.random import split


class NCA(nn.Module):
    n_layers: int
    d_state: int
    d_embd: int
    kernel_size: int = 3
    nonlin: str = 'gelu'

    def setup(self):
        layers = []
        for _ in range(self.n_layers):
            layers.extend([
                nn.Conv(features=self.d_embd, kernel_size=(self.kernel_size, self.kernel_size), feature_group_count=1),
                nn.LayerNorm(),
                getattr(nn, self.nonlin),
            ])
        layers.append(nn.Conv(features=self.d_state, kernel_size=(self.kernel_size, self.kernel_size)))
        self.dynamics_net = nn.Sequential(layers)
        
        self.obs_net = nn.Conv(features=3, kernel_size=(1, 1))

    def __call__(self, state):
        ds, obs = self.dynamics_net(state), self.obs_net(state)
        return ds, obs

def create_nca_step_fn(nca, nca_params, dt=0.01, p_drop=0.0, vid_type=None):
    def nca_step(carry, _):
        state, rng = carry
        dstate, obs = nca.apply(nca_params, state)
        rng, _rng = split(rng)
        drop_mask = jax.random.uniform(_rng, (H, W, 1)) < p_drop
        next_state = state + dt * dstate * (1. - drop_mask)
        next_state = next_state / jnp.linalg.norm(next_state, axis=-1, keepdims=True)
        obs_data = dict(state=state, obs=obs)
        return (next_state, rng), jax.tree.map(lambda x: obs_data[x], vid_type)
    return nca_step
    
def create_nca_rollout_fn(*args, **kwargs, n_steps):
    nca_step = create_nca_step_fn(*args, **kwargs)
    def nca_rollout(carry, ):
        return jax.lax.scan(nca_step, carry, length=n_steps)
    return nca_rollout

def sample_init_state(_rng, height=32, width=32, d_state=16, init_state="randn"):
    if init_state == "zeros":
        state = jnp.full((height, width, d_state), -1.)
    elif init_state == "point":
        state = jnp.full((height, width, d_state), -1.)
        state = state.at[height // 2, width // 2, :].set(1.)
    elif init_state == "randn":
        state = jax.random.normal(_rng, (height, width, d_state))
    else:
        raise NotImplementedError
    state = state / jnp.linalg.norm(state, axis=-1, keepdims=True)
    return state



