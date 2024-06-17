import torch
from torch import nn

from tqdm.auto import tqdm

class NCABlock(nn.Module):
    def __init__(self, d_embd, kernel_size=3, nonlin='GELU', padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(d_embd, d_embd, kernel_size=kernel_size, padding='same', padding_mode=padding_mode)
        self.nonlin = getattr(nn, nonlin)()
        
    def forward(self, x):
        x = self.conv(x)
        x = (x - x.mean(dim=-3, keepdim=True))/(x.std(dim=-3, keepdim=True) + 1e-8)  # layernorm
        x = self.nonlin(x)
        return x
        
class NCA(nn.Module):
    def __init__(self, n_layers, d_state, d_embd, kernel_size=3, nonlin='GELU', padding_mode='zeros'):
        super().__init__()
        self.dynamics_net = nn.Sequential()
        self.dynamics_net.append(nn.Conv2d(d_state, d_embd, kernel_size=1, padding='same', padding_mode=padding_mode))
        for _ in range(n_layers):
            self.dynamics_net.append(NCABlock(d_embd, kernel_size=kernel_size, nonlin=nonlin, padding_mode=padding_mode))
        self.dynamics_net.append(nn.Conv2d(d_embd, d_state, kernel_size=1, padding='same', padding_mode=padding_mode))
        self.obs_net = nn.Conv2d(d_state, 3, kernel_size=1, padding='same', padding_mode=padding_mode)
    
    def forward(self, x):
        return self.dynamics_net(x), self.obs_net(x)

class NCAWrapper(nn.Module):
    def __init__(self, n_layers, d_state, d_embd, kernel_size=3, nonlin='GELU', padding_mode='zeros', dt=0.01, p_drop=0., vid_type=None, n_steps=64):
        super().__init__()
        self.nca = NCA(n_layers, d_state, d_embd, kernel_size=kernel_size, nonlin=nonlin, padding_mode=padding_mode)
        self.dt, self.p_drop, self.vid_type = dt, p_drop, vid_type
        self.n_steps = n_steps
        
    def forward_step(self, state):
        B, D, H, W = state.shape
        dstate, obs = self.nca(state)
        drop_mask = torch.rand(1, H, W, dtype=state.dtype, device=state.device) < self.p_drop
        next_state = state + self.dt * dstate * (1. - drop_mask.to(state.dtype))
        next_state = next_state / next_state.norm(dim=-3, keepdim=True, p=2)
        # obs_data = dict(state=state, obs=obs)
        return next_state, obs
        
    def forward(self, state):
        for t in range(self.n_steps):
            state, _ = self.forward_step(state)
        _, obs = self.forward_step(state)
        return state, obs

def sample_init_state(height:int=224, width:int=224, d_state:int=16, bs:int=1, init_state:str ="randn",
                      device:torch.device=torch.device("cpu"), dtype:torch.dtype=torch.float):
    if init_state == "zeros":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
    elif init_state == "point":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
        state[:, :, height//2, width//2]  = 1.
    elif init_state == "randn":
        state = torch.randn((bs, d_state, height, width), device=device, dtype=dtype)
    else:
        raise NotImplementedError
    state = state / state.norm(dim=-3, keepdim=True, p=2)
    return state


if __name__ == "__main__":
    nca = NCAWrapper(2, 16, 32, n_steps=64)
    print(nca)
    print(sum(p.numel() for p in nca.parameters()))

    state = sample_init_state(224, 224, 16, 1)
    state = state.to('cuda')
    nca = nca.to('cuda')

    with torch.no_grad():
        for i in tqdm(range(1000)):
            state, _ = nca(state)

    nca = torch.jit.script(nca)  # Convert to ScriptModule
            
    with torch.no_grad():
        for i in tqdm(range(1000)):
            state, _ = nca(state)

