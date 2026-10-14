# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen GDN state binding for native GPU ModelRunnerV2 microbatches.

Native scheduling, cache allocation/COW, pre/postprocessing and DBO execution
remain upstream-owned. Only recurrent state identity crosses a microbatch split.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.mamba_utils import get_mamba_groups
from vllm.v1.worker.utils import AttentionGroup


class QwenGDNModelState(MambaHybridModelState):
    """Private state selected by the registered text-only AFD Qwen model."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._state_ids_by_req_slot: dict[int, torch.Tensor] = {}
        self._full_batch_state_ids: dict[int, torch.Tensor] | None = None

    @contextmanager
    def bind_full_batch(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        *,
        for_capture: bool,
    ) -> Iterator[None]:
        """Bind full-step IDs before native slicing; initialize fresh GDN state."""
        num_reqs = input_batch.num_reqs
        query_lens = (
            input_batch.query_start_loc[1 : num_reqs + 1]
            - input_batch.query_start_loc[:num_reqs]
        )
        initialize_state = input_batch.has_prefill and not for_capture
        fresh = (
            (input_batch.seq_lens[:num_reqs] - query_lens == 0) & (query_lens > 0)
            if initialize_state
            else None
        )
        full_batch_state_ids: dict[int, torch.Tensor] = {}
        for spec, group_ids in get_mamba_groups(kv_cache_config).items():
            if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
                continue
            for group_id in group_ids:
                if group_id in full_batch_state_ids:
                    continue
                # Derive from the supplied full batch, including capture's dummy
                # tables. A stage-local seq_len may name an earlier/null block.
                physical_ids = mamba_get_block_table_tensor(
                    block_tables[group_id][:num_reqs],
                    input_batch.seq_lens[:num_reqs],
                    spec,
                    self.cache_config.mamba_cache_mode,
                )[:, 0]
                if group_id not in self._state_ids_by_req_slot:
                    self._state_ids_by_req_slot[group_id] = torch.empty(
                        self.max_num_reqs,
                        dtype=physical_ids.dtype,
                        device=self.device,
                    )
                slot_ids = self._state_ids_by_req_slot[group_id]
                slot_ids.fill_(NULL_BLOCK_ID)
                slot_ids.index_copy_(
                    0,
                    input_batch.idx_mapping[:num_reqs],
                    physical_ids,
                )
                full_batch_state_ids[group_id] = slot_ids
                if fresh is not None:
                    # A fresh prefill can become one decode token in stage 0.
                    # Native decode reads state directly, and native attention
                    # block zeroing excludes Mamba. Initialize once, before both
                    # stages; continuations/prefix hits target reserved NULL0.
                    fresh_ids = torch.where(fresh, physical_ids, NULL_BLOCK_ID).long()
                    forward_context = (
                        self.vllm_config.compilation_config.static_forward_context
                    )
                    for layer_name in kv_cache_config.kv_cache_groups[
                        group_id
                    ].layer_names:
                        for state in forward_context[layer_name].kv_cache:
                            state.index_fill_(0, fresh_ids, 0)
        previous = self._full_batch_state_ids
        self._full_batch_state_ids = full_batch_state_ids
        try:
            yield
        finally:
            self._full_batch_state_ids = previous

    # Patch reason: pinned hybrid preparation rejects DBO, selects builder 0 and
    # derives GDN IDs from stage-truncated seq_lens, losing the full-step target.
    # Patch functionality: preserve native metadata/lifecycle, select each native
    # stage builder, and bind its existing state-ID storage to full-step targets.
    # Signature: matches MambaHybridModelState.prepare_attn; no added parameters.
    # Upstream: vllm/v1/worker/gpu/model_states/mamba_hybrid.py,
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607 (vLLM 0.30.0).
    # Removal plan: upstream hybrid DBO with full-step recurrent state identity.
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        # ### PATCH START: keep non-DBO preparation native.
        if self._full_batch_state_ids is None:
            return super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
                ubatch_idx,
            )
        # ### PATCH END: keep non-DBO preparation native.
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                # A row is a spec-decode row only when its whole prompt is already
                # computed, i.e. exactly one non-draft (decode) token is scheduled.
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    # ### PATCH START: use this stage's native builder.
                    builder = group.get_metadata_builder(ubatch_idx)
                    # ### PATCH END: use this stage's native builder.
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                ctx = self._ensure_align_ctx(
                    kv_cache_config, mamba_group_ids, block_tables
                )
                all_group_indices = ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            # ### PATCH START: select isolated native stage buffers.
            ubatch_idx=ubatch_idx,
            # ### PATCH END: select isolated native stage buffers.
        )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata,
                attn_groups,
                for_capture=for_capture,
            )
        # ### PATCH START: bind both pieces to the full-step recurrent state.
        for group_id, groups in enumerate(attn_groups):
            if group_id not in self._full_batch_state_ids:
                continue
            for group in groups:
                builder = group.get_metadata_builder(ubatch_idx)
                if not isinstance(builder, GDNAttentionMetadataBuilder):
                    continue
                # A group's layer names share one native metadata object.
                metadata = attn_metadata[group.layer_names[0]]
                if self.cache_config.mamba_cache_mode == "none" and (
                    cudagraph_mode != CUDAGraphMode.FULL
                ):
                    # The sole none-mode block is already the full-step target.
                    # Eager kernels trim to num_actual_tokens, so zero-query
                    # padding does no work. Preserve the borrowed table/aliases.
                    continue
                state_ids = metadata.non_spec_state_indices_tensor
                assert state_ids is not None
                bound_ids = torch.full_like(state_ids, NULL_BLOCK_ID)
                real_reqs = input_batch.num_reqs
                query_lens = (
                    input_batch.query_start_loc[1 : real_reqs + 1]
                    - input_batch.query_start_loc[:real_reqs]
                )
                full_ids = self._full_batch_state_ids[group_id][
                    input_batch.idx_mapping[:real_reqs]
                ]
                bound_ids[:real_reqs] = torch.where(
                    query_lens > 0,
                    full_ids,
                    NULL_BLOCK_ID,
                )
                # FULL metadata already references the builder's persistent
                # stage buffer. Preserve its address and fill all padded rows.
                state_ids.copy_(bound_ids)
        # ### PATCH END: bind both pieces to the full-step recurrent state.
        return attn_metadata
