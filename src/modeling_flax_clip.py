# coding=utf-8
# Copyright 2021 The OpenAI Team Authors, The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
# from ...modeling_flax_utils import (
#     ACT2FN,
#     FlaxPreTrainedModel,
#     append_replace_return_docstrings,
#     overwrite_call_docstring,
# )
# from ...utils import ModelOutput, add_start_docstrings, logging
# from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from transformers.utils import ModelOutput, add_start_docstrings, logging
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

from einops import rearrange


class FlaxCLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (embed_dim,))

        # self.patch_embedding = nn.Conv(
        #     embed_dim,
        #     kernel_size=(patch_size, patch_size),
        #     strides=(patch_size, patch_size),
        #     padding="VALID",
        #     use_bias=False,
        #     dtype=self.dtype,
        #     kernel_init=jax.nn.initializers.normal(),
        # )
        self.patch_embedding = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal())

        self.num_patches = (image_size // patch_size) ** 2
        num_positions = self.num_patches + 1
        self.position_embedding = nn.Embed(num_positions, embed_dim, embedding_init=jax.nn.initializers.normal())
        self.position_ids = jnp.expand_dims(jnp.arange(0, num_positions, dtype="i4"), axis=0)

    def __call__(self, pixel_values):
        # print(pixel_values.shape) # (16, 224, 224, 3)
        
        # patch_embeds = self.patch_embedding(pixel_values)
        pixel_values = rearrange(pixel_values, '... (H ph) (W pw) D -> ... H W (ph pw D)',
                                 ph=self.config.patch_size, pw=self.config.patch_size)
        patch_embeds = self.patch_embedding(pixel_values)
        # print(patch_embeds.shape) # (16, 7, 7, 768)

        
        batch_size, height, width, channels = patch_embeds.shape
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

        class_embeds = jnp.expand_dims(self.class_embedding, axis=(0, 1))
        class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class FlaxCLIPAttention(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = self.config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

        self.causal = isinstance(self.config, CLIPTextConfig)
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype="i4"))

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        causal_attention_mask = None
        if self.causal:
            query_length, key_length = query.shape[1], key.shape[1]
            causal_attention_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_mask = combine_masks(attention_mask, causal_attention_mask, dtype="i4")
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxCLIPMLP(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.activation_fn = ACT2FN[self.config.hidden_act]
        self.fc1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = FlaxCLIPAttention(self.config, dtype=self.dtype)
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.mlp = FlaxCLIPMLP(self.config, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += attn_outputs[1:]

        return outputs


class FlaxCLIPLayerCollection(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxCLIPEncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxCLIPEncoder(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = FlaxCLIPLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        inputs_embeds,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )



class FlaxCLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxCLIPVisionEmbeddings(self.config, dtype=self.dtype)
        self.pre_layrnorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        self.post_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FlaxCLIPVisionModule(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vision_model = FlaxCLIPVisionTransformer(self.config, dtype=self.dtype)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


if __name__ == "__main__":
    import jax.numpy as jnp
    from tqdm.auto import tqdm
    import copy

    print('loading jax')
    rng = jax.random.PRNGKey(0)
    print('done')
    img = jax.random.uniform(rng, (5, 224, 224, 3))

    from clip_jax import MyFlaxCLIPBackprop
    clip = MyFlaxCLIPBackprop()

    def loss_fn(x):
        y = clip.embed_img(x)
        return y.mean()
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    loss_fn = jax.jit(loss_fn)

    # for i in tqdm(range(1000)):
    #     y = loss_fn(img[0])
    # for i in tqdm(range(1000)):
    #     y, grad = grad_fn(img[0])


    # print(dir(model))
    # config = model.config.vision_config
    # print(config)
    # config = CLIPVisionConfig()
    # net = FlaxCLIPVisionModule(config)



    # print(model.module.vision_model.embeddings)

    # print(dir(model))



    # print(jax.tree.map(lambda x: x.shape, model.params['vision_model']['embeddings']['patch_embedding']))

    # config = CLIPVisionConfig()
    # net = FlaxCLIPVisionModule(config)
    # x = jnp.zeros((16, 224, 224, 3))
    # rng = jax.random.PRNGKey(0)
    # params = net.init(rng, x)
    # print(jax.tree.map(lambda x: x.shape, params['params']['vision_model']['embeddings']['patch_embedding']))

    # params = copy.copy(model.params)
    # params['vision_model']['embeddings']['patch_embedding']['kernel'] = params['vision_model']['embeddings']['patch_embedding']['kernel'].reshape(32*32*3, 768)
    # print(jax.tree.map(lambda x: x.shape, model.params['vision_model']['embeddings']['patch_embedding']))


    # for a, b in zip(jax.tree.flatten(model.params), jax.tree.flatten(params)):
        # print(a.shape, b.shape, a.shape==b.shape)















