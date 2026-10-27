# ------------------------------------------------------------------------------
# Copyright (c) lucidrains and its affiliates.
# To view a copy of this license, visit
# https://github.com/lucidrains/x-transformers/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Hongyu LIU
# ------------------------------------------------------------------------------
from einops import rearrange, repeat, pack, unpack
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import Module
from x_transformers.attend import Attend, Intermediates
from x_transformers.x_transformers import l2norm, AttentionLayers, ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding, LayerNorm, LayerIntermediates
from functools import partial, wraps
from loguru import logger
from torch import Tensor, cat, arange, einsum
from torch.nn import Module, ModuleDict
from typing import Union, Optional
from dataclasses import dataclass
from functools import partial
import numpy as np
import math


@dataclass
class LayerIntermediates:
    hiddens:            Optional[list[Tensor]] = None   # all hiddens, before the final norm (in pre-norm architecture)
    last_hidden:        Optional[Tensor] = None         # very last hidden after all attention layers, after the final norm
    attn_intermediates: Optional[list[Intermediates]] = None
    layer_hiddens:      Optional[list[Tensor]] = None
    attn_z_loss:        Optional[Tensor] = None
    mems:               Optional[Tensor] = None
    last_layer_hiddens: Optional[Tensor] = None
    attn_pooled_tokens: Optional[Tensor] = None
    memory_tokens:      Optional[Tensor] = None
    logit_entropies:    Optional[Tensor] = None
    logits:             Optional[Tensor] = None
    cache_length:       int = 0

LinearNoBias = partial(nn.Linear, bias = False)

# helpers
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def first(it, default = None):
    return it[0] if len(it) > 0 else default

def is_empty(x):
    return len(x) == 0

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn = None):
    if not exists(fn):
        fn = identity

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def softclamp(t, value):
    return (t / value).tanh() * value

def masked_mean(t, mask = None, dim = 1):
    if not exists(mask):
        return t.mean(dim = dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim).clamp(min = 1.)
    return num / den

def pad_at_dim(t, pad: tuple, dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# entropy

def calc_entropy(
    t: Tensor,
    is_prob = False
):
    prob = t.softmax(dim = -1) if not is_prob else t
    return -(prob * log(prob)).sum(dim = -1)

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: list,
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = loss.sum(dim=1)

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = tuple(d.pop(key) for key in  keys)
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    prefix_len = len(prefix)
    kwargs_without_prefix = {key[prefix_len:]: value for key, value in kwargs_with_prefix.items()}
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2
class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.emb.weight)

class TransformerWrapper(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: dict[str, int] = dict(),
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        tie_embedding = False,
        logits_dim = None,
        return_only_embed = False,
        num_output_heads = 1,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        recycling = False,            # from Jumper et al. - Alphafold2
        train_max_recycle_steps = 4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient = 1.,       # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
        average_pool_embed = False,
        use_cls_token = False,
        num_cls_tokens = 1,
        attn_pool = False,
        num_pooled_tokens = 1,
        attn_pool_depth = 1,
        dim_pooled_tokens = None,
        squeeze_out_last_dim = False,
        token_emb: Optional[TokenEmbedding] = None,
        mixture_of_softmax = False,
        mixture_of_softmax_k = 4,
        sigsoftmax_logits = False,
        ff_deep_embed = False,
        to_logits: Optional[Module] = None,
        add_continuous_pred_head = False
    ):
        super().__init__()

        dim = attn_layers.dim
        depth = attn_layers.depth

        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_cls_tokens = num_cls_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed

        if not exists(token_emb):
            token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        self.token_emb = token_emb

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        # additional embeddings - say type embedding from BERT

        self.embeds = None

        if len(embed_num_tokens) > 0:
            self.embeds = ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # deep embed

        # credit goes to Braden Koszarsky for first devising value embeddings in nanogpt-speedrun project
        # then Bo Peng for coming up with this alternate design in feedforward for RWKV 8
        # improvements were clearest to me (on my toy setup) with multiplying on output of feedforward, will try with attention at future date

        self.ff_deep_embed = None
        if ff_deep_embed:
            self.ff_deep_embed = nn.Parameter(torch.ones(num_tokens, depth, dim))

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        assert num_output_heads > 0

        assert at_most_one_of(average_pool_embed, use_cls_token)

        # maybe recycling

        self.recycling = recycling
        self.recycled_proj = LinearNoBias(dim, dim) if recycling else None

        self.train_max_recycle_steps = train_max_recycle_steps

        # either cls token or attn pool, but not both

        assert not (use_cls_token and attn_pool)

        # classic cls token from the bert days

        self.cls_token = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, dim))
            nn.init.normal_(self.cls_token, std = 0.02)

        # attn pool

        self.attn_pool = None

        if attn_pool:
            self.attn_pool = AttentionPool(dim = default(dim_pooled_tokens, dim), dim_context = dim, num_pooled_tokens = num_pooled_tokens, depth = attn_pool_depth, heads = self.attn_layers.attn_heads, dim_head = self.attn_layers.attn_dim_head)

        # whether to average pool the embed (`global average pool`)

        self.average_pool_embed = average_pool_embed

        # output type

        self.output_is_log_prob = mixture_of_softmax

        self.to_mixture = None
        self.combine_mixture = None

        if mixture_of_softmax:
            assert num_output_heads == 1

            self.to_mixture = Sequential(
                LinearNoBias(dim, dim * mixture_of_softmax_k),
                Rearrange('... (k d) -> ... k d', k = mixture_of_softmax_k)
            )

            self.combine_mixture = LinearNoBias(dim, mixture_of_softmax_k)

        # sig softmax

        self.sigsoftmax_logits = sigsoftmax_logits

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, num_tokens)

        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif tie_embedding:
            assert isinstance(token_emb, TokenEmbedding), 'can only tie embedding if using `TokenEmbedding`'
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.to_logits = ModuleList([LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)])
        else:
            self.to_logits = LinearNoBias(dim, logits_dim) if not exists(to_logits) else to_logits

        # add a head that predicts the embedding of the next step

        self.add_continuous_pred_head = add_continuous_pred_head

        if add_continuous_pred_head:

            self.to_next_embed_pred = nn.Sequential(
                LinearNoBias(dim, dim),
                nn.SiLU(),
                LinearNoBias(dim, dim)
            )

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # squeeze out last dimension if possible

        self.squeeze_out_last_dim = squeeze_out_last_dim

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0 and not recycling and self.attn_layers.can_cache_kv
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if hasattr(self.token_emb, 'init_'):
            self.token_emb.init_()

        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)

    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        return_embeddings_and_intermediates = False,
        return_logit_entropies = False,
        return_next_embed_pred = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        input_not_include_cache = False,
        token_emb_kwargs = dict(),
        to_logits_kwargs = dict(),
        **kwargs,
    ):

        # if sequence is None, auto create an empty one if `prepend_embeds` was supplied

        if not exists(x):
            assert exists(prepend_embeds)
            x = prepend_embeds.new_empty((prepend_embeds.shape[0], 0), dtype = torch.long)

        # shapes and variables

        b, n, device, token_ids, num_mems, has_memory_tokens, emb_frac_gradient, orig_mask = x.shape[0], x.shape[1], x.device, x, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient, mask

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss | return_embeddings_and_intermediates
        return_embeddings = return_embeddings | (not exists(self.to_logits)) | return_embeddings_and_intermediates

        # take care of position embedding offsets in the presence of cache and sequence is less than cache length (not full sequence)

        seq_pos_offset = 0

        if exists(cache) and input_not_include_cache:
            seq_pos_offset = cache.cache_length

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos, offset = seq_pos_offset) if not external_pos_emb else pos
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        # add additional embeddings

        assert not (exists(self.embeds) ^ (len(embed_ids) > 0)), '`embed_num_tokens` must be defined on `TransformerWrapper`'

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                mask = cat((prepend_mask, mask), dim = -1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # init embed

        init_embed = x

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe deep embeds

        deep_embed_and_ids = None

        if exists(self.ff_deep_embed):
            deep_embed_and_ids = (self.ff_deep_embed, token_ids)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, '... -> b ...', b = b)
            x, cls_packed_shape = pack([cls_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value = True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim = -2, value = 0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        # attn layers kwargs

        kwargs = dict(
            **kwargs,
            seq_pos_offset = seq_pos_offset,
            seq_start_pos = seq_start_pos,
            input_not_include_cache = input_not_include_cache
        )

        # attention layers
        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, 'you did not train with recycling'

            # regular

            attended, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, deep_embeds_and_ids = deep_embed_and_ids, return_hiddens = True, **kwargs)

        else:
            # recycling

            recycle_steps = default(recycle_steps, (randrange(self.train_max_recycle_steps) + 1) if self.training else None)
            assert exists(recycle_steps) and recycle_steps > 0, '`recycle_steps` must be provided on forward if recycling is turned on and not training'

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = self.recycled_proj(attended.detach()) if not first_step else 0.

                    attended, intermediates = self.attn_layers(x + maybe_recycled, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, **kwargs)

        x = attended
        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            x = x[:, :mem_seq]

        # store last layer hiddens, for access in case of cls token or attention pooling

        intermediates.last_layer_hiddens = x

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        # cls token(s)

        if exists(self.cls_token):
            x, last_layer_hiddens = unpack(x, cls_packed_shape, 'b * d')

            intermediates.last_layer_hiddens = last_layer_hiddens

            if x.shape[1] == 1:
                x = rearrange(x, 'b 1 d -> b d')  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # attention pool

        is_encoder = not self.attn_layers.causal
        return_pooled_tokens = exists(self.attn_pool) and is_encoder

        if (
            exists(self.attn_pool) and
            (return_intermediates or is_encoder) # in a new paper, they use attention pooling on decoder - so we'll default to returning pooled tokens if encoder, but for decoder, they must set `return_intermediates`
        ):

            attn_pooled_tokens = self.attn_pool(x, mask = mask)

            intermediates.attn_pooled_tokens = attn_pooled_tokens

        # handle expansion to mixture if needed (for mixture of softmax)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim = -1)
            x = self.to_mixture(x)

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        # maybe sig softmax

        if self.sigsoftmax_logits:
            logits = logits + logits.sigmoid().log()

        # handle maybe combine mixture

        if exists(combine_mixture):
            with autocast('cuda', enabled = False):
                prob = logits.softmax(dim = -1)
                mos = einsum('... k d, ... k -> ... d', prob, combine_mixture)
                logits = log(mos)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple((rearrange(t, '... 1 -> ...') if t.shape[-1] == 1 else t) for t in cast_tuple(logits))

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns
        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings_and_intermediates:
            out = (x, intermediates)
        elif return_embeddings:
            out = x
        elif return_pooled_tokens:
            intermediates.logits = logits
            out = attn_pooled_tokens
        else:
            out = logits

        # maybe next embed pred

        if return_next_embed_pred:
            assert self.add_continuous_pred_head
            next_embed_out = self.to_next_embed_pred(x)

            out = (out, (next_embed_out, init_embed))

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in  intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = [cat(pair, dim = -2) for pair in zip(mems, hiddens)] if exists(mems) else hiddens
            new_mems = [t[..., -self.max_mem_len:, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out
