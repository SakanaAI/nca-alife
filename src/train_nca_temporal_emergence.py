import numpy as np

from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

import torch
from torch import nn
from torchvision import transforms
from torch.nn.functional import cross_entropy

from PIL import Image
from transformers import AutoProcessor, CLIPModel

import matplotlib.pyplot as plt
import util
import argparse
from collections import defaultdict

from models_torch import NCA, sample_init_state
from clip_torch import MyTorchCLIP
from MSOEmultiscale import MyOpticalFlowNet

import imageio
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--dtype", type=str, default='float32')
group.add_argument("--device", type=str, default='cuda:0')

group = parser.add_argument_group("model")
group.add_argument("--img_size", type=int, default=64)
group.add_argument("--d_state", type=int, default=16)
group.add_argument("--perception", type=str, default='gradient')
group.add_argument("--kernel_size", type=int, default=3)

group.add_argument("--init_state", type=str, default="point")
group.add_argument("--padding_mode", type=str, default="zeros")
group.add_argument("--dt", type=float, default=0.01)
group.add_argument("--dropout", type=float, default=0.5)

group = parser.add_argument_group("data")
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group.add_argument("--prompts", type=str, default="atom,leaf,cell,human;compound,tree,animal,civilization")
group.add_argument("--spatial_scales", type=str, default="0.1;1.0")
group.add_argument("--optical_flow_mag", type=float, default=1.)

group.add_argument("--coef_alignment", type=float, default=1.)
group.add_argument("--coef_optical_flow", type=float, default=0.)
group.add_argument("--coef_temporal_softmax", type=float, default=0.)
group.add_argument("--coef_spatial_softmax", type=float, default=0.)
group.add_argument("--coef_temporal_novelty", type=float, default=0.)
group.add_argument("--coef_spatial_novelty", type=float, default=0.)

group = parser.add_argument_group("optimization")
group.add_argument("--rollout_steps", type=int, default=64)
group.add_argument("--bptt_steps", type=int, default=16)
group.add_argument("--pool_size", type=int, default=1024)

group.add_argument("--bs", type=int, default=8)
group.add_argument("--lr", type=float, default=3e-4)
group.add_argument("--n_iters", type=int, default=10000)
group.add_argument("--clip_grad_norm", type=float, default=1.)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    print(args)
    util.save_pkl(args.save_dir, 'args', args)
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    

    def init_pool():
        state = sample_init_state(args.img_size, args.img_size, args.d_state, args.pool_size, args.init_state, device=device, dtype=dtype)
        time = torch.randint(0, args.rollout_steps, (args.pool_size, ), device=device)
        return dict(state=state, time=time)
    pool = init_pool()

    flow_net = MyOpticalFlowNet(device=device, dtype=dtype)
    clip_model = MyTorchCLIP(args.clip_model, device=device, dtype=dtype)
    # prompts = args.prompt.split(';')
    # z_text = clip_model.embed_text(prompts) # (P, D)

    spatial_scales = [float(s) for s in args.spatial_scales.split(';')]
    prompts = [p.split(',') for p in args.prompts.split(';')]
    z_text_all = rearrange([clip_model.embed_text(p.split(',')) for p in args.prompts.split(';')], "S T D -> S T D")
    n_scales, n_times, _ = z_text_all.shape
    assert args.rollout_steps%n_times == 0
    seg_len = args.rollout_steps//n_times

    augment_img = transforms.Compose([
        # transforms.RandomRotation()
        transforms.RandomPerspective(distortion_scale=0.5, p=1., fill=0.),
    ])
    crop_img_scales = [transforms.RandomResizedCrop(224, scale=(s, s), ratio=(3./4., 4./3.)) for s in spatial_scales]
    opt_flow_resize = transforms.Resize(128)

    # ------------------------ Model ------------------------
    nca = NCA(d_state=args.d_state, perception=args.perception, kernel_size=args.kernel_size, padding_mode=args.padding_mode,
                 d_embds=[48, 128], cell_norm=False, state_unit_norm=True, dt=args.dt, dropout=args.dropout).to(device, dtype)
    # nca = torch.jit.script(nca)
    opt = torch.optim.AdamW(nca.parameters(), lr=args.lr, weight_decay=0.)
    num_params = sum(p.numel() for p in nca.parameters())
    print(f"# of parameters: {num_params}")
    print(f"Image size: {args.img_size}x{args.img_size}x3={args.img_size*args.img_size*3}")
    print(f"Image Compression: {num_params/(args.img_size*args.img_size*3)}")
    print(f"Video Compression: {num_params/(args.rollout_steps*args.img_size*args.img_size*3)}")
    n_epochs = args.n_iters * args.bs * args.bptt_steps / (args.pool_size * args.rollout_steps)
    print(f"Estimated # of epochs: {n_epochs:.2f}")

    time2grad = {}
    def save_grad(t):
        def hook(grad):
            time2grad[t] = grad
        return hook

    def nca_unroll_batch():
        vid = []
        idx = torch.randperm(args.pool_size)[:args.bs]
        state, time = pool['state'][idx], pool['time'][idx]
        for t in range(args.bptt_steps):
            state, obs = nca.forward_step(state)
            vid.append(obs)
            reset_mask = ((time+t)%args.rollout_steps) == 0
            init_state = sample_init_state(args.img_size, args.img_size, args.d_state, args.bs, args.init_state,
                                           state_unit_norm=True, device=device, dtype=dtype)
            state[reset_mask] = init_state[reset_mask]
            if t%(args.bptt_steps//8)==0:
                state.register_hook(save_grad(t))
        pool['state'][idx], pool['time'][idx] = state.detach(), ((time+args.bptt_steps)%args.rollout_steps)
        vid = rearrange(vid, "T B D H W -> B T D H W").sigmoid()
        time_end = (pool['time'][idx] - 1)%args.rollout_steps # final timestep for the observation video
        return vid, time_end

    def loss_fn():
        vid, time_end = nca_unroll_batch()
        img = vid[:, -1]
        img = rearrange([augment_img(crop_img(img)) for crop_img in crop_img_scales], "S B D H W -> (S B) D H W")
        z_img = rearrange(clip_model.embed_img(img), "(S B) D -> B S D", S=n_scales)
        # z_img: B S D     # z_text_all: St Tt D
        label = time_end//seg_len # (B, )

        loss_dict = {}
        # ------------------------ Loss Alignment ------------------------
        loss_dict['loss_alignment'] = -rearrange(z_img, "B S D -> B S 1 D") @ rearrange(z_text_all[:, label, :], "St B D -> B St D 1")
        # ------------------------ Loss Temporal Softmax ------------------------
        logits = rearrange(z_img, "B S D -> B S 1 1 D") @ rearrange(z_text_all, "St Tt D -> St Tt D 1") # B S Tt 1 1
        loss_dict['loss_temporal_softmax'] = cross_entropy(rearrange(logits, "B S Tt 1 1 -> (B S) Tt"), repeat(label, "B -> (B S)", S=n_scales), reduction='none')
        # ------------------------ Loss Spatial Softmax ------------------------
        logits = rearrange(z_img, "B S D -> B S 1 1 D") @ rearrange(z_text_all[:, label, :], "St B D -> B 1 St D 1") # B S St 1 1
        label_scale = torch.arange(n_scales, device=device)
        loss_dict['loss_spatial_softmax'] = cross_entropy(rearrange(logits, "B S St 1 1 -> (B S) St"), repeat(label_scale, "S -> (B S)", B=args.bs), reduction='none')
        # ------------------------ Loss Temporal Novelty ------------------------
        loss_dict['loss_temporal_novelty'] = rearrange(z_img, "B S D -> S B D") @ rearrange(z_img, "B S D -> S D B") # S B B
        # ------------------------ Loss Spatial Novelty ------------------------
        loss_dict['loss_spatial_novelty'] = rearrange(z_img, "B S D -> B S D") @ rearrange(z_img, "B S D -> B D S") # B S S
        # ------------------------ Loss Optical Flow ------------------------
        if args.coef_optical_flow > 0:
            optical_flow = flow_net.get_optical_flow(opt_flow_resize(vid[:, -2]), opt_flow_resize(vid[:, -1])) # B 2 H W
            loss_dict['loss_optical_flow'] = (optical_flow.norm(dim=-3).mean(dim=(-1, -2))-args.optical_flow_mag).abs()

        loss = 0.
        for k in loss_dict:
            loss_dict[k] = loss_dict[k].mean()
            coef = getattr(args, k.replace('loss_', 'coef_'))
            loss = loss + coef * loss_dict[k].mean()
        loss_dict['loss'] = loss
        return loss_dict
    
    @torch.no_grad
    def create_full_video(bs=1):
        vid = []
        state = sample_init_state(args.img_size, args.img_size, args.d_state, bs, args.init_state, device=device, dtype=dtype)
        for t in range(int(args.rollout_steps*1.5)):
            state, obs = nca.forward_step(state)
            vid.append(obs)
        vid = rearrange(vid, "T B D H W -> B T H W D").sigmoid()  # NOTE: B T H W D instead of B T D H W
        return vid

    data = defaultdict(list)
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        opt.zero_grad()

        loss_dict = loss_fn()
        loss_dict['loss'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.clip_grad_norm)
        opt.step()

        if i_iter%10== 0:
            [data[k].append(v.detach()) for k, v in loss_dict.items()]
            data['grad_norm'].append(grad_norm)
            data['grad_bptt'].append(torch.stack([time2grad[k].norm(dim=-3).mean() for k in sorted(list(time2grad.keys()))]))
            pbar.set_postfix(loss=loss_dict['loss'].item())

        if args.save_dir is not None and (i_iter%(args.n_iters//5)==0 or i_iter==args.n_iters-1):
            vid = create_full_video(bs=1)[0] # T H W D
            vid = (vid*255).to(torch.uint8).cpu().numpy()

            util.save_pkl(args.save_dir, 'vid', vid)
            imageio.mimwrite(f'{args.save_dir}/vid.mp4', vid, fps=30, codec='libx264')
            imageio.mimwrite(f'{args.save_dir}/vid.gif', vid, fps=30)
            
            util.save_pkl(args.save_dir, 'data', {k: torch.stack(v).cpu().numpy() for k, v in data.items()})
            torch.save(nca.state_dict(), f"{args.save_dir}/nca.pt")

            plt.figure(figsize=(10, 5))
            plt.subplot(211); plt.plot(torch.stack(data['loss']).cpu().numpy())
            plt.subplot(212); plt.imshow(rearrange(vid[::(vid.shape[0]//8), :, :, :], "T H W D -> (H) (T W) D"))
            plt.savefig(f'{args.save_dir}/overview_{i_iter:06d}.png')
            plt.close()

    
if __name__ == "__main__":
    main(parse_args())
    

