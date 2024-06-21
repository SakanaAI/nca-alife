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

from models_torch import NCAWrapper, sample_init_state

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
group.add_argument("--kernel_size", type=int, default=3)
group.add_argument("--nonlin", type=str, default="GELU")

group.add_argument("--init_state", type=str, default="point")
group.add_argument("--padding_mode", type=str, default="zeros")
group.add_argument("--dt", type=float, default=0.01)
group.add_argument("--p_drop", type=float, default=0.0)

group.add_argument("--rollout_steps", type=int, default=64)
group.add_argument("--bptt_steps", type=int, default=64)

group.add_argument("--pool_size", type=int, default=1024)

group = parser.add_argument_group("data")
group.add_argument("--target_img_path", type=str, default=None)
group.add_argument("--prompt", type=str, default="a red apple;a green apple")
group.add_argument("--n_augs", type=int, default=4)
group.add_argument("--augs", type=str, default="crop+pers")  # crop+pers+jitter

group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1)
group.add_argument("--lr", type=float, default=1e-3)
group.add_argument("--n_iters", type=int, default=10000)
group.add_argument("--clip_grad_norm", type=int, default=1.)



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



    torch.randint(0, args.rollout_steps, args.pool_size)
    pool = dict(state=torch, times=times)


    

    # ------------------------ Objective ------------------------
    if args.target_img_path is not None:
        target_img = Image.open(args.target_img_path).convert('RGB').resize((args.img_size, args.img_size))
        target_img = torch.from_numpy(np.array(target_img)).to(device, dtype)/255.
        target_img = rearrange(target_img, "H W D -> D H W")

        def obj_loss_fn(x):
            return ((x - target_img)**2).mean()
    else:
        model = CLIPModel.from_pretrained(f"openai/{args.clip_model}")
        processor = AutoProcessor.from_pretrained(f"openai/{args.clip_model}")
        model = model.to(device, dtype)
    
        inputs = processor(text=args.prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            z_text = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            z_text = z_text/z_text.norm(dim=-1, keepdim=True)

        img_mean = torch.tensor(processor.image_processor.image_mean, device=device, dtype=dtype)
        img_std = torch.tensor(processor.image_processor.image_std, device=device, dtype=dtype)
        print("img_mean: ", img_mean.tolist())
        print("img_std: ", img_std.tolist())

        def obj_loss_fn(x):
            x = (x-img_mean[:, None, None])/img_std[:, None, None]
            z_img = model.get_image_features(x)
            z_img = z_img/z_img.norm(dim=-1, keepdim=True)
            loss = -(z_img@z_text.T).mean()
            return loss

    # ------------------------ Model ------------------------
    if args.substrate == 'img':
        if args.init == 'randn':
            dofs = torch.randn((3, args.img_size, args.img_size), device=device, dtype=dtype).requires_grad_()
        else:
            dofs = torch.full((3, args.img_size, args.img_size), fill_value=0.5, device=device, dtype=dtype).requires_grad_()
        opt = torch.optim.AdamW([dofs], lr=args.lr, weight_decay=0.)
        def create_image():
            return repeat(torch.sigmoid(dofs), 'D H W -> B H W D', B=args.bs)
    else:
        nca = NCAWrapper(args.n_layers, args.d_state, args.d_embd, 
                         kernel_size=args.kernel_size, nonlin=args.nonlin, padding_mode=args.padding_mode,
                         dt=args.dt, p_drop=args.p_drop, n_steps=args.n_steps).to(device, dtype)
        nca = torch.jit.script(nca)
        opt = torch.optim.AdamW(nca.parameters(), lr=args.lr, weight_decay=0.)
        
        def create_image():
            state = sample_init_state(args.img_size, args.img_size, args.d_state, args.bs, args.init_state, device=device, dtype=dtype)
            state, obs = nca(state)
            return rearrange(torch.sigmoid(obs), 'B D H W -> B H W D')
    
    def loss_fn():
        x = create_image()
        # print('create image: ', x.shape)
        x = rearrange(x, "B H W D -> B D H W")
        # print('rearrange: ', x.shape)
        x = augment_img(x)
        # print('augment: ', x.shape)
        loss = obj_loss_fn(x)
        # print('loss: ', loss.shape)
        return loss


    
    losses = []
    pbar = tqdm(range(args.n_iters))
    for i in pbar:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        opt.step()
        
        losses.append(loss.detach())
        if i%1==0:
            pbar.set_postfix(loss=loss.item())

    if args.save_dir is not None:
        img = create_image()[0].detach().cpu().numpy().clip(0, 1)
        losses = torch.stack(losses).detach().cpu().numpy()
        
        util.save_pkl(args.save_dir, 'losses', losses)
        util.save_pkl(args.save_dir, 'img', img)
    
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(losses)
        plt.subplot(122)
        plt.imshow(img)
        plt.savefig(f'{args.save_dir}/overview.png')
        plt.close()

        if args.substrate=='nca':
            torch.save(nca.state_dict(), f"{args.save_dir}/nca.pt")


if __name__ == "__main__":
    main(parse_args())
    

