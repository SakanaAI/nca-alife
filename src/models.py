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

    def forward_dynamics(self, state):
        return self.dynamics_net(state)

    def forward_obs(self, state):
        return self.obs_net(state)

    def __call__(self, state):
        ds, obs = self.dynamics_net(state), self.obs_net(state)
        return ds, obs


def nca_rollout(nca, nca_params, _rng, state, rollout_steps=1000, dt=0.01, p_drop=0.0, vid_type='obs'):
    H, W, D = state.shape

    def forward_step(state, _rng):
        if vid_type == 'obs':
            obs = nca.apply(nca_params, state, method=nca.forward_obs)
        elif vid_type == 'state':
            obs = state
        elif vid_type == 'none':
            obs = None
        else:
            raise NotImplementedError

        dstate = nca.apply(nca_params, state, method=nca.forward_dynamics)
        drop_mask = jax.random.uniform(_rng, (H, W, 1)) < p_drop
        next_state = state + dt * dstate * (1. - drop_mask)
        next_state = next_state / jnp.linalg.norm(next_state, axis=-1, keepdims=True)
        return next_state, obs

    state, vid = jax.lax.scan(forward_step, state, split(_rng, rollout_steps))
    return state, vid


def sample_init_state(_rng, height=32, width=32, d_state=16, init_state="point"):
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
