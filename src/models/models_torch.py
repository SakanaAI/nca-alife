import torch
from torch import nn

from tqdm.auto import tqdm

from einops import repeat

class CellNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x): # B, D, H, W
        return (x - x.mean(dim=-3, keepdim=True))/(x.std(dim=-3, keepdim=True) + 1e-8)  # layernorm over dim=-3

def cell_dropout(x, p_drop=0.5):
    B, D, H, W = x.shape
    keep_mask = torch.rand(B, 1, H, W, dtype=x.dtype, device=x.device) < (1.-p_drop)
    return x * keep_mask.to(x.dtype)

class NCAPerceive(nn.Module):
    def __init__(self, padding_mode='circular'):
        super().__init__()
        self.padding_mode = padding_mode
    def forward(self, x):
        eye_kernel = torch.tensor([[0.,  0.,  0.], [0.,  1.,  0.], [0.,  0.,  0.]], device=x.device, dtype=x.dtype)
        dx = torch.tensor([[-1.,  0.,  1.], [-2.,  0.,  2.], [-1.,  0.,  1.]], device=x.device, dtype=x.dtype)/8.
        w = torch.stack([eye_kernel, dx, dx.T])
        w = repeat(w, 'K H W -> (16 K) 1 H W')
        x = nn.functional.pad(x, (1, 1, 1, 1), mode='constant' if self.padding_mode=='zeros' else self.padding_mode)
        x = nn.functional.conv2d(x, w, padding='valid', groups=16)
        return x
        
class NCA(nn.Module):
    # Note: Cell_norm makes it very slow...
    def __init__(self, d_state=16, # input vars
                 perception='gradient', kernel_size=3, padding_mode='zeros', # perceive vars
                 d_embds=[48, 128], cell_norm=False, # network vars
                 state_unit_norm=False, dt=0.01, dropout=0.5): # dynamics vars
        super().__init__()
        self.state_unit_norm, self.dt, self.dropout = state_unit_norm, dt, dropout
        
        if perception=='gradient':
            perceive = NCAPerceive(padding_mode=padding_mode)
        elif perception=='learned':
            perceive = nn.Conv2d(d_state, d_embds[0], kernel_size=kernel_size, padding='same', padding_mode=padding_mode,
                                 groups=d_state, bias=False)
        elif perception=='fullconv':
            perceive = nn.Conv2d(d_state, d_embds[0], kernel_size=kernel_size, padding='same', padding_mode=padding_mode)

        self.dynamics_net = nn.Sequential(perceive)
        for d_in, d_out in zip(d_embds[:-1], d_embds[1:]):
            self.dynamics_net.extend([
                nn.Conv2d(d_in, d_out, kernel_size=1),
                CellNorm() if cell_norm else nn.Identity(),
                nn.GELU(),
            ])
        self.dynamics_net.append(nn.Conv2d(d_embds[-1], d_state, kernel_size=1))
        self.obs_net = nn.Conv2d(d_state, 3, kernel_size=1)
        
    def forward_step(self, state):
        B, D, H, W = state.shape
        dstate, obs = self.dynamics_net(state), self.obs_net(state)
        next_state = state + self.dt * cell_dropout(dstate, self.dropout)
        if self.state_unit_norm:
            next_state = next_state / next_state.norm(dim=-3, keepdim=True)
        return next_state, obs

# def sample_init_state(height:int=224, width:int=224, d_state:int=16, bs:int=1, init_state:str ="randn",
                      # device:torch.device=torch.device("cpu"), dtype:torch.dtype=torch.float):
def sample_init_state(height=256, width=256, d_state=16, bs=1, init_state="randn", state_unit_norm=True,
                      device=None, dtype=None):
    if init_state == "zeros":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
    elif init_state == "point":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
        state[:, :, height//2, width//2]  = 1.
    elif init_state == "pointgrid":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
        start, spacing = height//16, height//8
        state[:, :, start::spacing, start::spacing]  = 1.
    elif init_state == "randn":
        state = torch.randn((bs, d_state, height, width), device=device, dtype=dtype)
    elif init_state == "circle":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
        center, radius = (height//2, width//2), height//8
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        distance = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        circle_mask = distance <= radius
        for i in range(bs):
            state[i, 0][circle_mask] = 1.0
    elif init_state == "circle_rand":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
        center, radius = (height//2, width//2), height//8
        radius = radius * torch.rand((), device=device, dtype=dtype)
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        distance = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        circle_mask = distance <= radius
        for i in range(bs):
            j = torch.randint(0, d_state, ())
            state[i, j][circle_mask] = torch.randn((), device=device, dtype=dtype)
    elif init_state == "square":
        state = torch.zeros((bs, d_state, height, width), device=device, dtype=dtype) - 1.
        state[:, 0, height//8*3:height//8*5, width//8*3:width//8*5] = 1.
    else:
        raise NotImplementedError

    if state_unit_norm:
        state = state / state.norm(dim=-3, keepdim=True, p=2)
    return state


if __name__ == "__main__":
    nca = NCA(2, 16, 32, n_steps=64)
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

