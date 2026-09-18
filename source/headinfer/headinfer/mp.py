import transformers
from transformers.models.llama.modeling_llama import (
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaSdpaAttention,
    LlamaFlashAttention2,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2FlashAttention2
)
import torch
from flash_attn import flash_attn_varlen_func

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache, QuantizedCacheConfig, HQQQuantizedCache
from transformers.modeling_flash_attention_utils  import _flash_attention_forward


def _flash_attention_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    dropout: float,
    is_causal: bool,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Small patchable wrapper for layer-flat ragged decode attention."""
    window_size = (
        (sliding_window, sliding_window)
        if sliding_window is not None and max_seqlen_k > sliding_window
        else (-1, -1)
    )
    return flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout,
        causal=is_causal,
        window_size=window_size,
        deterministic=False,
    )


def LlamaAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) :
    output_attentions = False

    bsz, q_len, hd = hidden_states.size()
    chunk_size = hd // self.num_key_value_heads
    num_heads = self.num_heads // self.num_key_value_heads

    if not hasattr(self, 'q_proj_list'):
        self.q_proj_list = list((self.q_proj.weight.split(self.head_dim * num_heads, dim=0)))
        # self.q_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'k_proj_list'):
        self.k_proj_list = list((self.k_proj.weight.split(self.head_dim, dim=0)))
        # self.k_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'v_proj_list'):
        self.v_proj_list = list((self.v_proj.weight.split(self.head_dim, dim=0)))
        # self.v_proj.weight.data.storage().resize_(0)


    attn_output_list = [None for _ in range((self.num_key_value_heads))]

    # The original HeadInfer path consumes one KV group immediately.  Layer-
    # batched sparse selection and layer-level Full GQA execution both require
    # all projected Q/K/V groups before consuming the cache.
    prepared_group_states = None
    use_layer_batched_selection = (
        q_len == 1
        and past_key_value is not None
        and getattr(past_key_value, "layer_batched_selection", False)
        and hasattr(past_key_value, "prepare_layer_selection")
    )
    use_layer_full_attention = (
        q_len == 1
        and past_key_value is not None
        and getattr(past_key_value, "layer_full_attention_execution", False)
    )
    if use_layer_batched_selection or use_layer_full_attention:
        prepared_group_states = []
        layer_projection = bool(getattr(self, '_twilight_layer_projection', False)) and use_layer_batched_selection
        layer_rope = bool(getattr(self, '_twilight_layer_rope', False)) and use_layer_batched_selection
        if layer_projection:
            # Full-weight views were retained before head-wise prefill mutates
            # Parameter.data. No per-token weight concatenation or copy.
            for projection, weight in zip(
                (self.q_proj, self.k_proj, self.v_proj), self._twilight_full_projection_weights
            ):
                projection.weight.data = weight
            query_states = self.q_proj(hidden_states).view(
                bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(
                bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(
                bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            for i in range(self.num_key_value_heads):
                prepared_group_states.append((
                    query_states[:, i*num_heads:(i+1)*num_heads],
                    key_states[:, i:i+1], value_states[:, i:i+1], cos, sin))
        for i in range(0 if layer_projection else self.num_key_value_heads):
            self.q_proj.weight.data = self.q_proj_list[i].data
            self.k_proj.weight.data = self.k_proj_list[i].data
            self.v_proj.weight.data = self.v_proj_list[i].data
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(
                bsz, q_len, num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, 1, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, 1, self.head_dim
            ).transpose(1, 2)
            if position_embeddings is None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            if not layer_rope:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            prepared_group_states.append(
                (query_states, key_states, value_states, cos, sin)
            )
        if layer_rope and not layer_projection:
            cos, sin = prepared_group_states[0][3:]
            all_query = torch.cat([state[0] for state in prepared_group_states], dim=1)
            all_key = torch.cat([state[1] for state in prepared_group_states], dim=1)
            all_query, all_key = apply_rotary_pos_emb(all_query, all_key, cos, sin)
            prepared_group_states = [
                (all_query[:, i*num_heads:(i+1)*num_heads], all_key[:, i:i+1], state[2], cos, sin)
                for i, state in enumerate(prepared_group_states)
            ]
        if use_layer_batched_selection:
            past_key_value.prepare_layer_selection(
                entries=[self.layer_idx + i for i in range(self.num_key_value_heads)],
                queries=[state[0] for state in prepared_group_states],
            )

    use_twilight_gqa_group = (
        prepared_group_states is not None
        and getattr(past_key_value, "gqa_groupwise_execution", False)
        and hasattr(past_key_value, "update_layer_gqa_group_ragged")
    )
    if use_twilight_gqa_group:
        if attention_mask is not None:
            raise NotImplementedError(
                "Twilight GQA group-wise decode currently requires no attention mask"
            )
        entries = [self.layer_idx + i for i in range(self.num_key_value_heads)]
        flat_keys, flat_values, cu_seqlens_k, max_seqlen_k, _ = (
            past_key_value.update_layer_gqa_group_ragged(
                key_states=[state[1] for state in prepared_group_states],
                value_states=[state[2] for state in prepared_group_states],
                queries=[state[0] for state in prepared_group_states],
                entries=entries,
            )
        )
        group_queries = torch.stack(
            [state[0][0, :, 0] for state in prepared_group_states], dim=0
        )
        group_count = int(group_queries.shape[0])
        cu_seqlens_q = torch.arange(
            group_count + 1, dtype=torch.int32, device=group_queries.device
        )
        flat_output = _flash_attention_varlen_forward(
            group_queries,
            flat_keys,
            flat_values,
            cu_seqlens_q,
            cu_seqlens_k,
            1,
            max_seqlen_k,
            dropout=0.0,
            is_causal=self.is_causal,
            sliding_window=getattr(self, "sliding_window", None),
        )
        attn_output = flat_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    use_layer_flat_ragged = (
        prepared_group_states is not None
        and getattr(past_key_value, "layer_flat_ragged_execution", False)
        and hasattr(past_key_value, "update_layer_flat_ragged")
    )
    if use_layer_flat_ragged:
        if attention_mask is not None:
            raise NotImplementedError(
                "layer-flat ragged decode currently requires no attention mask"
            )
        entries = [self.layer_idx + i for i in range(self.num_key_value_heads)]
        flat_keys, flat_values, cu_seqlens_k, max_seqlen_k, _ = (
            past_key_value.update_layer_flat_ragged(
                key_states=[state[1] for state in prepared_group_states],
                value_states=[state[2] for state in prepared_group_states],
                queries=[state[0] for state in prepared_group_states],
                entries=entries,
            )
        )
        flat_queries = torch.cat(
            [state[0][0, :, 0] for state in prepared_group_states], dim=0
        ).unsqueeze(1)
        query_heads = int(flat_queries.shape[0])
        cu_seqlens_q = torch.arange(
            query_heads + 1, dtype=torch.int32, device=flat_queries.device
        )
        flat_output = _flash_attention_varlen_forward(
            flat_queries,
            flat_keys,
            flat_values,
            cu_seqlens_q,
            cu_seqlens_k,
            1,
            max_seqlen_k,
            dropout=0.0,
            is_causal=self.is_causal,
            sliding_window=getattr(self, "sliding_window", None),
        )
        attn_output = flat_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    if use_layer_full_attention:
        if prepared_group_states is None:
            raise AssertionError("layer-level Full execution is missing projections")
        if not hasattr(past_key_value, "update_layer_full"):
            raise TypeError("layer-level Full cache is missing update_layer_full()")
        entries = [self.layer_idx + i for i in range(self.num_key_value_heads)]
        layer_key_states, layer_value_states = past_key_value.update_layer_full(
            key_states=[state[1] for state in prepared_group_states],
            value_states=[state[2] for state in prepared_group_states],
            entries=entries,
        )
        layer_queries = torch.cat(
            [state[0].transpose(1, 2) for state in prepared_group_states], dim=2
        )
        attn_output = _flash_attention_forward(
            layer_queries,
            layer_key_states,
            layer_value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value
    
    for i in range(self.num_key_value_heads):
        bsz, q_len, hd = hidden_states.size()

        if prepared_group_states is not None:
            query_states, key_states, value_states, cos, sin = prepared_group_states[i]
        else:
            self.q_proj.weight.data = self.q_proj_list[i].data
            self.k_proj.weight.data = self.k_proj_list[i].data
            self.v_proj.weight.data = self.v_proj_list[i].data

            # print(hidden_states.shape, self.q_proj.weight.shape)

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                # Dynamic sparse caches need the post-RoPE Query to rank
                # historical pages.  Existing caches intentionally ignore
                # unknown cache kwargs, so this is backward compatible.
                "query_states": query_states,
            }
            cache_entry = self.layer_idx + i
            key_states, value_states = past_key_value.update(
                key_states, value_states, cache_entry, cache_kwargs
            )
            query_head_lengths = (
                past_key_value.get_last_query_head_lengths(cache_entry)
                if hasattr(past_key_value, "get_last_query_head_lengths")
                else None
            )
        else:
            query_head_lengths = None
    
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    
        attention_kwargs = {
            "position_ids": position_ids,
            "dropout": 0.0,
            "sliding_window": getattr(self, "sliding_window", None),
            "use_top_left_mask": self._flash_attn_uses_top_left_mask,
            "is_causal": self.is_causal,
        }
        if query_head_lengths is not None and (
            len(set(query_head_lengths)) > 1
            or getattr(
                past_key_value,
                "per_head_varlen_attention_reference",
                False,
            )
        ):
            if attention_mask is not None:
                raise NotImplementedError(
                    "variable per-Query-head cache lengths require decode without an attention mask"
                )
            per_head_outputs = []
            for query_head, valid_length in enumerate(query_head_lengths):
                if getattr(
                    past_key_value,
                    "per_head_varlen_attention_reference",
                    False,
                ):
                    cu_q = torch.tensor(
                        [0, 1], dtype=torch.int32, device=query_states.device
                    )
                    cu_k = torch.tensor(
                        [0, valid_length],
                        dtype=torch.int32,
                        device=query_states.device,
                    )
                    output = _flash_attention_varlen_forward(
                        query_states[:, :, query_head].reshape(1, 1, self.head_dim),
                        key_states[:, :valid_length, query_head].reshape(
                            valid_length, 1, self.head_dim
                        ),
                        value_states[:, :valid_length, query_head].reshape(
                            valid_length, 1, self.head_dim
                        ),
                        cu_q,
                        cu_k,
                        1,
                        valid_length,
                        dropout=0.0,
                        is_causal=self.is_causal,
                        sliding_window=getattr(self, "sliding_window", None),
                    ).reshape(1, 1, 1, self.head_dim)
                else:
                    output = _flash_attention_forward(
                        query_states[:, :, query_head : query_head + 1],
                        key_states[:, :valid_length, query_head : query_head + 1],
                        value_states[:, :valid_length, query_head : query_head + 1],
                        None,
                        q_len,
                        **attention_kwargs,
                    )
                per_head_outputs.append(output)
            attn_output = torch.cat(per_head_outputs, dim=2)
        else:
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                **attention_kwargs,
            )
    
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output_list[i] = attn_output
        
    attn_output = torch.cat(attn_output_list, dim=-1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def Qwen2Attention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) :
    output_attentions = False

    bsz, q_len, hd = hidden_states.size()
    chunk_size = hd // self.num_key_value_heads
    num_heads = self.num_heads // self.num_key_value_heads

    if not hasattr(self, 'q_proj_list'):
        self.q_proj_list = list((self.q_proj.weight.split(self.head_dim * num_heads, dim=0)))
        self.q_proj_bias_list = list((self.q_proj.bias.split(self.head_dim * num_heads, dim=0)))
        # self.q_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'k_proj_list'):
        self.k_proj_list = list((self.k_proj.weight.split(self.head_dim, dim=0)))
        self.k_proj_bias_list = list((self.k_proj.bias.split(self.head_dim, dim=0)))
        # self.k_proj.weight.data.storage().resize_(0)
    if not hasattr(self, 'v_proj_list'):
        self.v_proj_list = list((self.v_proj.weight.split(self.head_dim, dim=0)))
        self.v_proj_bias_list = list((self.v_proj.bias.split(self.head_dim, dim=0)))
        # self.v_proj.weight.data.storage().resize_(0)


    attn_output_list = [None for _ in range((self.num_key_value_heads))]
    
    for i in range(self.num_key_value_heads):
        bsz, q_len, hd = hidden_states.size()

        self.q_proj.weight.data = self.q_proj_list[i].data
        self.q_proj.bias.data = self.q_proj_bias_list[i].data
        self.k_proj.weight.data = self.k_proj_list[i].data
        self.k_proj.bias.data = self.k_proj_bias_list[i].data
        self.v_proj.weight.data = self.v_proj_list[i].data
        self.v_proj.bias.data = self.v_proj_bias_list[i].data

        # print(hidden_states.shape, self.q_proj.weight.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
    
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "query_states": query_states,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx + i, cache_kwargs)
    
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
    
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output_list[i] = attn_output
        
    attn_output = torch.cat(attn_output_list, dim=-1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def mp_headinfer(model):

    if isinstance(model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        
        layer_idx = 0
        for idx, layer in enumerate(model.model.layers):
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            module = layer.self_attn
            module.layer_idx = layer_idx
            layer_idx += module.num_key_value_heads
            module.head_group = 8
    elif isinstance(model, transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM):
        Qwen2FlashAttention2.forward = Qwen2Attention_fast_forward
        
        layer_idx = 0
        for idx, layer in enumerate(model.model.layers):
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            module = layer.self_attn
            module.layer_idx = layer_idx
            layer_idx += module.num_key_value_heads
            module.head_group = 4

    print("[OK] HeadInfer Patched Successfully.")


def mp_simulate_decode(model, start_length=1024000):
    past_key_values = OffloadedCache()
    config = model.config

    for idx, layer in enumerate(model.model.layers):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        module = layer.self_attn

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        # head_dim = config.hidden_size
        cache_shape = (1, 1, start_length, module.head_dim)
        key_states = torch.zeros(cache_shape, dtype=dtype, device=device)
        value_states = torch.zeros(cache_shape, dtype=dtype, device=device)

        for i in range(config.num_key_value_heads):
            cache_kwargs = {'full_head': None}
            key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx + i, cache_kwargs)

    print("[OK] Simulation of Decoding Ready to start.")
    return past_key_values
