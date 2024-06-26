import numpy as np

from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

import torch
from torch import nn
from torchvision import transforms

from PIL import Image
from transformers import AutoProcessor, CLIPModel

import matplotlib.pyplot as plt
import util
import argparse
from collections import defaultdict

from models_torch import NCA, sample_init_state

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--dtype", type=str, default='float32')
group.add_argument("--device", type=str, default='cuda:0')

group = parser.add_argument_group("model")
group.add_argument("--img_size", type=int, default=64)
group.add_argument("--n_layers", type=int, default=2)
group.add_argument("--d_state", type=int, default=16)
group.add_argument("--d_embd", type=int, default=32)
group.add_argument("--locality", type=int, default=1)
group.add_argument("--kernel_size", type=int, default=3)
group.add_argument("--nonlin", type=str, default="GELU")

group.add_argument("--init_state", type=str, default="point")
group.add_argument("--padding_mode", type=str, default="zeros")
group.add_argument("--dt", type=float, default=0.01)
group.add_argument("--p_drop", type=float, default=0.0)

group.add_argument("--rollout_steps", type=int, default=64)
group.add_argument("--bptt_steps", type=int, default=16)

group.add_argument("--pool_size", type=int, default=1024)

group = parser.add_argument_group("data")
group.add_argument("--target_img_path", type=str, default=None)
# group.add_argument("--prompt", type=str, default="a red apple;a green apple;a blue apple;a yellow apple")
# group.add_argument("--prompt", type=str, default="a red apple;a green tree;a fat cat;the yellow sun")
group.add_argument("--prompt", type=str, default="a green tree")
group.add_argument("--n_augs", type=int, default=1)
group.add_argument("--augs", type=str, default="crop+pers")  # crop+pers+jitter

group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group.add_argument("--coef_alignment", type=float, default=1.)
group.add_argument("--coef_softmax", type=float, default=0.)
group.add_argument("--coef_temporal", type=float, default=0.)

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=8)
group.add_argument("--lr", type=float, default=1e-3)
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    trans_aug = []
    if 'crop' in args.augs:
        trans_aug.append( transforms.RandomResizedCrop(224, scale=(0.4, 1.)), )
    if 'pers' in args.augs:
        trans_aug.append( transforms.RandomPerspective(distortion_scale=0.5, p=1., fill=0.), )
    if 'jitter' in args.augs:
        trans_aug.append(transforms.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=0.1))
    trans_aug = transforms.Compose(trans_aug)
    def augment_img(x):
        xs = torch.cat([trans_aug(x) for i in range(args.n_augs)])
        return xs

    def init_pool():
        states = sample_init_state(args.img_size, args.img_size, args.d_state, args.pool_size, args.init_state, device=device, dtype=dtype)
        times = torch.randint(0, args.rollout_steps, (args.pool_size, ), device=device)
        return dict(states=states, times=times)
    pool = init_pool()

    # ------------------------ Objective ------------------------
    if args.target_img_path is not None:
        assert False
        target_img = Image.open(args.target_img_path).convert('RGB').resize((args.img_size, args.img_size))
        target_img = torch.from_numpy(np.array(target_img)).to(device, dtype)/255.
        target_img = rearrange(target_img, "H W D -> D H W")

        def obj_loss_fn(x):
            return ((x - target_img)**2).mean()
    else:
        clip_model = CLIPModel.from_pretrained(f"openai/{args.clip_model}")
        processor = AutoProcessor.from_pretrained(f"openai/{args.clip_model}")
        clip_model = clip_model.to(device, dtype)
        for p in clip_model.parameters():
            p.requires_grad = False

        prompts = args.prompt.split(';')
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            z_text = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            z_text = z_text/z_text.norm(dim=-1, keepdim=True)

        img_mean = torch.tensor(processor.image_processor.image_mean, device=device, dtype=dtype)
        img_std = torch.tensor(processor.image_processor.image_std, device=device, dtype=dtype)
        print("img_mean: ", img_mean.tolist())
        print("img_std: ", img_std.tolist())

        assert args.rollout_steps%len(prompts)==0
        n_segs, seg_len = len(prompts), args.rollout_steps//len(prompts)
        def obj_loss_fn(img, time, vid):
            label = time//seg_len
            x = (img-img_mean[:, None, None])/img_std[:, None, None]
            z_img = clip_model.get_image_features(x)
            z_img = z_img/z_img.norm(dim=-1, keepdim=True)

            # z_img: (Batch, D), z_text: (Prompts, D), label: (Batch, )

            loss_alignment = -(z_img*z_text[label]).sum(dim=-1).mean()
            
            logits = (z_img @ z_text.T) * clip_model.logit_scale.exp()  # Batch, Prompts
            loss_softmax = torch.nn.functional.cross_entropy(logits, label, reduction='none').mean()

            f_nxt, f_now = vid[:, 1:, :, :, :], vid[:, :-1, :, :, :]
            loss_temporal = -((f_nxt-f_now)**2).mean()

            loss = loss_alignment * args.coef_alignment + loss_softmax * args.coef_softmax + loss_temporal * args.coef_temporal
            return dict(loss=loss, loss_alignment=loss_alignment, loss_softmax=loss_softmax, loss_temporal=loss_temporal)

    # ------------------------ Model ------------------------
    nca = NCA(args.n_layers, args.d_state, args.d_embd, 
              locality=args.locality, kernel_size=args.kernel_size, nonlin=args.nonlin, padding_mode=args.padding_mode,
              dt=args.dt, p_drop=args.p_drop, n_steps=args.rollout_steps).to(device, dtype)
    # nca = torch.jit.script(nca)
    opt = torch.optim.AdamW(nca.parameters(), lr=args.lr, weight_decay=0.)
    num_params = sum(p.numel() for p in nca.parameters())
    print(f"# of parameters: {num_params}")
    print(f"Image size: {args.img_size}x{args.img_size}x3={args.img_size*args.img_size*3}")
    print(f"Image Compression: {num_params/(args.img_size*args.img_size*3)}")

    time2grad = {}

    def forward_chunk(state, time):
        def save_grad(t):
            def hook(grad):
                time2grad[t] = grad
            return hook

        vid = []
        for t in range(args.bptt_steps):
            state, obs = nca.forward_step(state)
            obs_time = time
            vid.append(obs)
            
            time += 1
            init_states = sample_init_state(args.img_size, args.img_size, args.d_state, args.bs, args.init_state, device=device, dtype=dtype)
            state[time>=args.rollout_steps] = init_states[time>=args.rollout_steps]
            time[time>=args.rollout_steps] = 0
            
            if t%(args.bptt_steps//8)==0:
                state.register_hook(save_grad(t))
                
        vid = torch.stack(vid, dim=-4)
        return state, time, obs, obs_time, vid
        
    def forward_batch():
        idx = torch.randperm(args.pool_size)[:args.bs]
        states, times = pool['states'][idx], pool['times'][idx]
        states, times, obs, obs_times, vid = forward_chunk(states, times)
        pool['states'][idx], pool['times'][idx] = states.detach(), times

        obs = rearrange(torch.sigmoid(obs), 'B D H W -> B D H W')
        vid = rearrange(torch.sigmoid(vid), 'B T D H W -> B T D H W')
        return obs, obs_times, vid
    
    @torch.no_grad
    def create_full_video():
        vid = []
        state = sample_init_state(args.img_size, args.img_size, args.d_state, args.bs, args.init_state, device=device, dtype=dtype)
        for t in range(args.rollout_steps):
            state, obs = nca.forward_step(state)
            vid.append(obs)
        return rearrange(torch.sigmoid(torch.stack(vid)), 'T B D H W -> B T H W D')

    def loss_fn():
        x, times, vid = forward_batch()
        x = augment_img(x)
        return obj_loss_fn(x, times, vid)

    losses = defaultdict(list)
    grad_norms = []
    grad_vs_times = []
    pbar = tqdm(range(args.n_iters))
    for i in pbar:
        opt.zero_grad()

        loss = loss_fn()
        loss['loss'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.clip_grad_norm)
        opt.step()

        [losses[k].append(v.detach()) for k, v in loss.items()]
            
        grad_norms.append(grad_norm.item())
        grad_vs_times.append([time2grad[k].norm(dim=-3).mean().item() for k in sorted(list(time2grad.keys()))])

        if i%100==0:
            pbar.set_postfix(loss=loss['loss'].item())

        if args.save_dir is not None and (i%(args.n_iters//5)==0 or i==args.n_iters-1):
            vid = create_full_video().cpu().numpy()

            util.save_pkl(args.save_dir, 'vid', vid)
            util.save_pkl(args.save_dir, 'losses', {k: torch.stack(v).cpu().numpy() for k, v in losses.items()})
            util.save_pkl(args.save_dir, 'grad_norms', np.array(grad_norms))
            util.save_pkl(args.save_dir, 'grad_vs_times', np.array(grad_vs_times))
            torch.save(nca.state_dict(), f"{args.save_dir}/nca.pt")

            plt.figure(figsize=(10, 5))
            plt.subplot(211); plt.plot(torch.stack(losses['loss']).cpu().numpy())
            plt.subplot(212); plt.imshow(rearrange(vid[:1, ::(vid.shape[1]//8), :, :, :], "B T H W D -> (B H) (T W) D"))
            plt.savefig(f'{args.save_dir}/overview_{i:06d}.png')
            plt.close()
    
if __name__ == "__main__":
    main(parse_args())
    

