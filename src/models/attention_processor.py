# *************************************************************************
# Copyright (2024) Bytedance Inc.
#
# Copyright (2024) LightningDrag Authors 
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
# *************************************************************************

import math
import inspect
from importlib import import_module
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange 
from diffusers.utils import deprecate


class MatchAttnProcessor(nn.Module):

    def __init__(self, embed_dim, hidden_size, use_norm=False):
        super().__init__()

        self.pred_residual=None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "MatchAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        match_embeddings = None,
        uc_mask = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
       
        L, S = query.size(-2), key.size(-2)
        scale = None
        dropout_p=0.0
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        self.attn_map = query @ key.transpose(-2, -1) * scale_factor
        self.before_attn_map = self.attn_map.clone()
        if self.matcher is not None:
            attn_size = self.attn_map.shape[-1]
            if self.attn_map.shape[-2] * 2 == self.attn_map.shape[-1]:
                if self.hard_corr: 
                    corr_input = F.interpolate(
                        self.seg_corr.float().unsqueeze(1), size = (attn_size // 2, attn_size // 2), mode="nearest"
                    )
                    attn_size_sqrt = int(math.sqrt(attn_size // 2))
                    self.attn_map[:, :, :, attn_size // 2:] = torch.where(
                        corr_input == 1,
                        self.attn_map[:, :, :, attn_size // 2:],  # Keep original attention map values
                        torch.tensor(float('-inf'), dtype=self.attn_map.dtype, device=self.attn_map.device)  # Set to -1 where mask is 0
                    )

                else:
                    if attn_size != 8192:
                        corr_input = torch.cat([
                            self.attn_map[:, :, :, attn_size // 2:].clone().detach(),
                            F.interpolate(self.seg_corr.float().unsqueeze(1), size=(attn_size // 2, attn_size // 2), mode="nearest")
                        ], dim=1) # [b 9, attn_size // 2, attn_size // 2]
                        attn_size_sqrt = int(math.sqrt(attn_size // 2))
                        corr_input = rearrange(corr_input, 'b c (hs ws) (ht wt) -> b c hs ws ht wt', hs=attn_size_sqrt, ws=attn_size_sqrt, ht=attn_size_sqrt, wt=attn_size_sqrt)
                        
                        refined_seg_corr = self.matcher(corr_input, self.blocks)
                        temp_ = refined_seg_corr.max() 
                        refined_seg_corr = ((rearrange(refined_seg_corr, 'b c hs ws ht wt -> b c (hs ws) (ht wt)')))  - temp_ / 4

                        if refined_seg_corr.shape[0] == 2: # inference
                            refined_seg_corr[0, :, :, :] = 0
                        self.attn_map[:, :, :, attn_size//2: ] = self.attn_map[:, :, :, attn_size // 2:] + refined_seg_corr
                    else:
                        corr_input = torch.cat([
                            F.interpolate(self.attn_map[:, :, :, attn_size // 2:].clone().detach(), size = (32*32, 32*32), mode="bilinear", align_corners=False),
                            self.seg_corr.float().unsqueeze(1),
                        ], dim=1)
                        
                        corr_input = rearrange(corr_input, 'b c (hs ws) (ht wt) -> b c hs ws ht wt', hs=32, ws=32, ht=32, wt=32)
                        refined_seg_corr = self.matcher(corr_input, self.blocks)
                        temp_ = refined_seg_corr.max() 
                        refined_seg_corr = rearrange(refined_seg_corr, 'b c hs ws ht wt -> b c (hs ws) (ht wt)')  - temp_ / 4

                        if refined_seg_corr.shape[0] == 2:
                            refined_seg_corr[0, :, :, :] = 0
                        refined_seg_corr = F.interpolate(refined_seg_corr, size = (attn_size // 2, attn_size // 2), mode = 'bilinear', align_corners=False)
                        self.attn_map[:, : , :, attn_size // 2:] = self.attn_map[:, :, :, attn_size//2:] + refined_seg_corr
            
                    torch.cuda.empty_cache() 
                    del refined_seg_corr
 
        attn_weight = self.attn_map
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        hidden_states = attn_weight @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states