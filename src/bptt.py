import argparse
import os
import pickle

import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax.random import split
from tqdm.auto import tqdm
from einops import rearrange

import data_utils
import unroll_ed
from agents.regular_transformer import BCTransformer
from util import save_pkl, save_json, tree_stack
import util

jax_config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--load_ckpt", type=str, default=None)
group.add_argument("--save_dir", type=str, default=None)
group.add_argument("--save_ckpt", type=lambda x: x=='True', default=False)

group = parser.add_argument_group("data")
group.add_argument("--dataset_paths", type=str, default=None)
group.add_argument("--exclude_dataset_paths", type=str, default=None)
group.add_argument("--n_augs", type=int, default=0)
group.add_argument("--aug_dist", type=str, default="uniform")
group.add_argument("--nv", type=int, default=4096)
group.add_argument("--nh", type=int, default=131072)

group = parser.add_argument_group("optimization")
# group.add_argument("--n_iters_eval", type=int, default=100)
group.add_argument("--n_iters", type=int, default=100000)
group.add_argument("--bs", type=int, default=128)
# group.add_argument("--mini_bs", type=int, default=None)
group.add_argument("--lr", type=float, default=3e-4)
group.add_argument("--lr_schedule", type=str, default="constant")  # constant or cosine_decay
group.add_argument("--weight_decay", type=float, default=0.)
group.add_argument("--clip_grad_norm", type=float, default=1.)

group = parser.add_argument_group("model")
group.add_argument("--d_obs_uni", type=int, default=32)
group.add_argument("--d_act_uni", type=int, default=8)
group.add_argument("--n_layers", type=int, default=4)
group.add_argument("--n_heads", type=int, default=8)
group.add_argument("--d_embd", type=int, default=256)
group.add_argument("--ctx_len", type=int, default=128)  # physical ctx_len of transformer
# group.add_argument("--seq_len", type=int, default=512)  # how long history it can see
group.add_argument("--mask_type", type=str, default="causal")

group = parser.add_argument_group("rollout")
group.add_argument("--env_id", type=str, default=None)
group.add_argument("--n_envs", type=int, default=64)
group.add_argument("--n_iters_rollout", type=int, default=1000)
group.add_argument("--video", type=lambda x: x=='True', default=False)

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)  # set all "none" to None
    # if args.mini_bs is None:
        # args.mini_bs = args.bs
    # assert args.bs % args.mini_bs == 0
    return args





