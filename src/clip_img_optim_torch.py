import numpy as np

from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat

import torch
from torch import nn
from torchvision import transforms


from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

import matplotlib.pyplot as plt
import util
import argparse

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--dtype", type=str, default='float32')
group.add_argument("--device", type=str, default='cuda:0')

group = parser.add_argument_group("data")
group.add_argument("--prompt", type=str, default='a red apple')
group.add_argument("--n_augs", type=int, default=8)
group.add_argument("--aug_crop", type=lambda x: x=='True', default=False)
group.add_argument("--aug_perspective", type=lambda x: x=='True', default=False)
group.add_argument("--aug_color", type=lambda x: x=='True', default=False)

group = parser.add_argument_group("model")
group.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
# "openai/clip-vit-base-patch32"
# "openai/clip-vit-large-patch14"

group.add_argument("--lr", type=float, default=1e-3)
group.add_argument("--n_iters", type=int, default=10000)



def augment_image(_rng, img):
    H, W, D = img.shape
    assert H==W
    
    _rng1, _rng2 = split(_rng)
    crop_ratio = jax.random.uniform(_rng1, (2, ), minval=0.5, maxval=1.)
    crop_ratio = crop_ratio.at[1].set(crop_ratio[0])
    crop_loc = jax.random.uniform(_rng2, (2, ), minval=0., maxval=(1.-crop_ratio)*H)
    
    scale = 1./crop_ratio
    translation = -scale*crop_loc
    img = jax.image.scale_and_translate(img, (224, 224, 3), [0, 1], scale, translation, 'linear')
    return img


def main(args):
    print(args)
    torch.manual_seed(args.seed)
    
    model = CLIPModel.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = model.to(device, dtype)

    inputs = processor(text=args.prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        z_text = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        z_text = z_text/z_text.norm(dim=-1, keepdim=True)

    mean = torch.tensor(processor.image_processor.image_mean, device=device, dtype=dtype)
    std = torch.tensor(processor.image_processor.image_std, device=device, dtype=dtype)
    print(mean, std)

    trans_aug = []
    if args.aug_color:
        trans_aug.append(transforms.ColorJitter(brightness=.5, contrast=0.5, saturation=0.5, hue=0.1))
    if args.aug_crop:
        trans_aug.append( transforms.RandomResizedCrop(224, scale=(0.1, 1.)), )
    if args.aug_perspective:
        trans_aug.append( transforms.RandomPerspective(distortion_scale=0.5, p=1., fill=0.), )
    trans_aug = transforms.Compose(trans_aug )
    
    def augment_img(x):
        xs = torch.cat([trans_aug(x) for i in range(args.n_augs)])
        return xs
    
    img = torch.full((224, 224, 3), fill_value=0.5, device=device, dtype=dtype)
    img.requires_grad_()
    opt = torch.optim.AdamW([img], lr=args.lr, weight_decay=0.)

    def loss_fn():
        x = rearrange((img-mean)/std, 'H W D -> 1 D H W')
        x = augment_img(x)
        z_img = model.get_image_features(x)
        z_img = z_img/z_img.norm(dim=-1, keepdim=True)
        loss = -(z_img@z_text.T).mean()
        return loss

    losses = []
    pbar = tqdm(range(args.n_iters))
    for i in pbar:
        loss = loss_fn()
        losses.append(loss.detach())
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i%10==0:
            pbar.set_postfix(loss=loss.item())

    if args.save_dir is not None:
        util.save_pkl(args.save_dir, 'losses', torch.stack(losses).detach().cpu().numpy())
        util.save_pkl(args.save_dir, 'img', img.detach().cpu().numpy())
        
        losses = torch.stack(losses).detach().cpu().numpy()
        img = img.clip(0, 1).detach().cpu().numpy()
    
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(losses)
        plt.subplot(122)
        plt.imshow(img)
        plt.savefig(f'{args.save_dir}/overview.png')
        plt.close()


if __name__ == "__main__":
    main(parser.parse_args())

