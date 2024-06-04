import jax
import jax.numpy as jnp
from flax import linen as nn


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


def forward_nca(nca, nca_params, _rng, state, rollout_steps=1000, obs_every_k=10, dt=0.01, p_drop=0.0):
    H, W, D = state.shape
    
    def forward_step(state, _rng):
        dstate = nca.apply(nca_params, state, method=nca.forward_dynamics)
        drop_mask = jax.random.uniform(_rng, (H, W, 1)) < self.p_drop
        state = state + dt * dstate * (1.-drop_mask)
        state = state/jnp.linalg.norm(state, axis=-1, keepdims=True)
        return state, obs
    
    def forward_chunk(x, _):
        x, vid = jax.lax.scan(forward_step, x, jnp.arange(obs_every_k))
        return x, vid[-1]
    state, vid = jax.lax.scan(forward_chunk, state, jnp.arange(rollout_steps//obs_every_k))
    return state, vid



