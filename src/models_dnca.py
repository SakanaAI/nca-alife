
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split


class DNCANetwork(nn.Module):
    d_state: int = 16
    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, pad_width=1, mode='wrap')
        x = nn.Conv(features=48, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.Conv(features=128, kernel_size=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.d_state, kernel_size=(1, 1))(x)
        return x


class DNCA():
    def __init__(self, grid_size=64, d_state=16, n_groups=1, identity_bias=0., temperature=1.):
        self.grid_size = grid_size
        self.d_state, self.n_groups = d_state, n_groups
        self.dnca = DNCANetwork(d_state=d_state*n_groups)

        self.identity_bias, self.temperature = identity_bias, temperature

    def default_params(self, rng):
        rng, _rng = split(rng)
        color_map = jax.random.normal(_rng, (self.n_groups, self.d_state, 3)) # unconstrainted

        rng, _rng = split(rng)
        net_params = self.dnca.init(_rng, jnp.zeros((self.grid_size, self.grid_size, self.d_state*self.n_groups))) # unconstrainted

        rng, _rng = split(rng)
        init = jax.random.normal(_rng, (self.n_groups, self.d_state)) # unconstrainted

        rng, _rng = split(rng)
        identity_bias = jax.random.normal(_rng, ()) # unconstrainted
        return dict(color_map=color_map, net_params=net_params, init=init, identity_bias=identity_bias)
    
    def init_state(self, rng, params):
        init = repeat(params['init'], "G D -> H W G D", H=self.grid_size, W=self.grid_size)
        state = jax.random.categorical(rng, init, axis=-1)
        return state
    
    def step_state(self, rng, state, params):
        state_oh = jax.nn.one_hot(state, self.d_state)
        state_oh_f = rearrange(state_oh, "H W G D -> H W (G D)")
        logits = self.dnca.apply(params['net_params'], state_oh_f)
        logits = rearrange(logits, "H W (G D) -> H W G D", G=self.n_groups)
        
        # identity_bias = jax.nn.sigmoid(params['identity_bias'])*10
        next_state = jax.random.categorical(rng, (logits + state_oh*self.identity_bias)/self.temperature, axis=-1)
        return next_state
    
    def render_state(self, state, params, img_size=None):
        def get_color(color_map, state):
            return color_map[state]
        # color_map: G D 3 # state: H W G
        get_color = jax.vmap(get_color, in_axes=(0, 2))
        img = get_color(jax.nn.sigmoid(params['color_map']), state)
        img = img.mean(axis=0) # average over groups
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    dnca = DNCA(d_state=4, n_groups=2)
    params = dnca.default_params(rng)

    state = dnca.init_state(rng, params)
    # print(state.shape)
    # print(state[:4, :4])

    print(state.shape)
    state, _ = dnca.step_state(rng, state, params)
    print(state.shape)


    img = dnca.render_state(state, params)
    print(img.shape, img.min(), img.max(), img.dtype)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.savefig("./temp/dnca.png")
    plt.close()






