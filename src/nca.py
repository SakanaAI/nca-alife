import jax
import jax.numpy as jnp
from flax import linen as nn


# class NCAOriginal(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         d_embd = x.shape[-1]
#         x = nn.Dense(features=4 * d_embd)(x)
#         x = nn.gelu(x)
#         x = nn.Dense(features=d_embd)(x)
#         return x


class NCA(nn.Module):
    n_layers: int
    d_embd: int
    kernel_size: int = 3
    nonlin: str = 'gelu'
    p_drop: float = 0.0
    
    @nn.compact
    def __call__(self, _rng, xin):
        H, W, D_in = xin.shape
        x = xin
        for _ in range(self.n_layers):
            x = nn.Conv(features=self.d_embd, kernel_size=(self.kernel_size, self.kernel_size))(x)
            # x = nn.LayerNorm()(x)
            x = getattr(nn, self.nonlin)(x)
            # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = nn.Conv(features=D_in, kernel_size=(kernel_size, kernel_size))(x)
        x = nn.Conv(features=D_in, kernel_size=(1, 1))(x)
        mask = 1-(jax.random.uniform(_rng, (H, W)) < self.p_drop)
        return xin + x * mask[:, :, None]


rng = jax.random.PRNGKey(0)
nca = NCA(n_layers=3, d_embd=64)

x = jax.random.normal(rng, (32, 32, 3))
params = nca.init(rng, rng, x)

def forward_step(x, _):
    return nca.apply(params, rng, x)
# jax.lax.scan(x, x, range(100))


y = nca.apply(params, rng, x)
print(x.shape)
print(y.shape)




