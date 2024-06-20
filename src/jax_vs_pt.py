from tqdm.auto import tqdm

test_jax = True


if test_jax:
    from flax import linen as nn
    import jax
    import jax.numpy as jnp
    
    class ConvNetFlax(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.Conv(features=64, kernel_size=(1, 1), padding='SAME')(x)
            x = nn.gelu(x)
            x = nn.Conv(features=64, kernel_size=(1, 1), padding='SAME')(x)
            x = nn.gelu(x)
            x = nn.Conv(features=64, kernel_size=(1, 1), padding='SAME')(x)
            x = nn.gelu(x)
            x = nn.Conv(features=64, kernel_size=(1, 1), padding='SAME')(x)
            x = nn.gelu(x)
            x = nn.Conv(features=16, kernel_size=(1, 1), padding='SAME')(x)
            return x

    nca = ConvNetFlax()
    rng = jax.random.PRNGKey(0)
    state = jax.random.normal(rng, (8, 512, 512, 16))
    params = nca.init(rng, state[0])

    def forward_step(state, _):
        state = state + 0.01*jax.vmap(nca.apply, in_axes=(None, 0))(params, state)
        state = state/jnp.linalg.norm(state, axis=-1, keepdims=True)
        return state, None

    @jax.jit
    def forward_chunk(state, _):
        return jax.lax.scan(forward_step, state, jnp.arange(100))

    for i in tqdm(range(100)):
        state, _ = forward_chunk(state, None)
        
    

else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConvNetPytorch(nn.Module):
        def __init__(self):
            super(ConvNetPytorch, self).__init__()
            self.conv1 = nn.Conv2d(16, 64, kernel_size=(3, 3), padding='same')
            self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding='same')
            self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding='same')
            self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding='same')
            self.conv5 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding='same')
            self.conv6 = nn.Conv2d(64, 16, kernel_size=(1, 1), padding='same')
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.gelu(self.conv2(x))
            x = F.gelu(self.conv3(x))
            x = F.gelu(self.conv4(x))
            x = F.gelu(self.conv5(x))
            x = self.conv6(x)
            return x


    device = 'cuda'
    nca = ConvNetPytorch().to(device)
    state = torch.randn(8, 16, 512, 512).to(device)

    for i in tqdm(range(10000)):
        with torch.no_grad():
            state = state + 0.01*nca(state)
        state = state/torch.norm(state, dim=1, keepdim=True)




