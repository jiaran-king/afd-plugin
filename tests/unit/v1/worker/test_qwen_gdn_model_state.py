# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contracts against the installed pinned vLLM runtime.

Actual native slicing/GDN metadata and AFD methods execute on CPU tensors.
GPU-only constructors are bypassed; conv/chunk kernel metadata helpers are
stubbed. These tests do not establish CUDA kernel or graph replay correctness.
"""

from __future__ import annotations

import ast
import copy
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
assert vllm.__file__ is not None
pytestmark = pytest.mark.vllm_runtime
# Use installed exact runtime source unless the existing test-source override is set.
UPSTREAM = (
    Path(os.environ["AFD_TEST_VLLM_SOURCE"])
    if "AFD_TEST_VLLM_SOURCE" in os.environ
    else Path(vllm.__file__).resolve().parent.parent
)

MAX_REQS = 32
MAX_TOKENS = 160
BLOCK_SIZE = 16


@pytest.fixture
def api(monkeypatch):
    import vllm.v1.attention.backends.gdn_attn as gdn_module
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner
    from vllm.v1.worker.utils import AttentionGroup

    from afd_plugin.compat.patches import gpu_mrv2_ubatching as patch
    from afd_plugin.v1.worker.qwen_gdn_model_state import QwenGDNModelState

    monkeypatch.setattr(
        gdn_module, "compute_causal_conv1d_metadata", lambda *a, **k: (None, None, None)
    )
    return SimpleNamespace(
        CUDAGraphMode=CUDAGraphMode,
        GDNAttentionMetadataBuilder=GDNAttentionMetadataBuilder,
        MambaAttentionBackendEnum=MambaAttentionBackendEnum,
        MambaSpec=MambaSpec,
        InputBatch=InputBatch,
        InputBuffers=InputBuffers,
        UBatchRunner=UBatchRunner,
        AttentionGroup=AttentionGroup,
        QwenGDNModelState=QwenGDNModelState,
        prepare=patch.prepare,
        maybe_build_ubatch_runner=patch.maybe_build_ubatch_runner,
    )


def batch(api, query_lens, seq_lens, *, slots=None, padded=None, has_prefill=False):
    n = len(query_lens)
    total = sum(query_lens)
    padded = padded or total
    req_padded = padded if padded > total else n
    buffers = api.InputBuffers(MAX_REQS, MAX_TOKENS, torch.device("cpu"))
    out = api.InputBatch.make_dummy(n, total, buffers)
    offsets = np.zeros(req_padded + 1, dtype=np.int32)
    np.cumsum(query_lens, out=offsets[1 : n + 1])
    offsets[n + 1 :] = total
    buffers.query_start_loc[: req_padded + 1] = torch.from_numpy(offsets)
    buffers.seq_lens[:n] = torch.tensor(seq_lens, dtype=torch.int32)
    slots = slots or list(range(n))
    slot_array = np.zeros(req_padded, dtype=np.int32)
    slot_array[:n] = slots
    prefill = np.array([q > 1 for q in query_lens], dtype=np.bool_)
    return replace(
        out,
        num_reqs_after_padding=req_padded,
        num_tokens_after_padding=padded,
        idx_mapping=torch.tensor(slot_array, dtype=torch.int64),
        idx_mapping_np=slot_array,
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: req_padded + 1],
        query_start_loc_np=offsets,
        seq_lens=buffers.seq_lens[:req_padded],
        seq_lens_cpu_upper_bound=buffers.seq_lens[:req_padded].clone(),
        num_computed_tokens_np=np.full(n, 900, dtype=np.int32),
        is_prefilling_np=prefill,
        has_prefill=has_prefill,
        input_ids=buffers.input_ids[:padded],
        positions=buffers.positions[:padded],
        is_padding=buffers.is_padding[:padded],
    )


def setup(api, *, mode="align", groups=1, full=False, noncontiguous=False):
    spec = api.MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 4), (2, 3, 3)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=api.MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mode,
    )
    forward: dict[str, SimpleNamespace] = {}
    kv_groups = []
    attn = []
    state = object.__new__(api.QwenGDNModelState)
    state._state_ids_by_req_slot = {}
    state._full_batch_state_ids = None
    state.cache_config = SimpleNamespace(mamba_cache_mode=mode)
    state.device = torch.device("cpu")
    state.max_num_reqs = MAX_REQS
    state.max_model_len = 8192
    state._align_mode = mode == "align"
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(MAX_REQS, dtype=torch.int32)
    cfg = SimpleNamespace(
        num_speculative_tokens=0,
        cache_config=state.cache_config,
        compilation_config=SimpleNamespace(static_forward_context=forward),
    )
    state.vllm_config = cfg
    for gid in range(groups):
        names = [f"gdn{gid}.a", f"gdn{gid}.b"]
        for name in names:
            conv = torch.arange(64 * 3 * 4, dtype=torch.float32).reshape(64, 3, 4) + 1
            ssm = (
                torch.arange(64 * 2 * 3 * 3, dtype=torch.float32).reshape(64, 2, 3, 3)
                + 2
            )
            if noncontiguous:
                conv = conv.transpose(1, 2)
                ssm = ssm.transpose(2, 3)
            forward[name] = SimpleNamespace(kv_cache=(conv, ssm))
        kv_groups.append(SimpleNamespace(kv_cache_spec=spec, layer_names=names))
        builders = []
        for _ in range(2):
            b = object.__new__(api.GDNAttentionMetadataBuilder)
            b.kv_cache_spec = spec
            b.vllm_config = cfg
            b.use_spec_decode = False
            b.use_full_cuda_graph = full
            b.decode_cudagraph_max_bs = MAX_REQS
            b.non_spec_state_indices_tensor = torch.empty(MAX_REQS, dtype=torch.int32)
            b.non_spec_query_start_loc = torch.empty(MAX_REQS + 1, dtype=torch.int32)
            b._build_chunk_metadata = lambda *a: (None, None)
            builders.append(b)
        attn.append(
            [
                api.AttentionGroup(
                    backend=type("GDNBackend", (), {}),
                    layer_names=names,
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                    metadata_builders=builders,
                )
            ]
        )
    kv = SimpleNamespace(kv_cache_groups=kv_groups)
    state._get_mamba_group_info = lambda _: (list(range(groups)), spec)
    runner = object.__new__(api.UBatchRunner)
    runner.model_state = state
    runner.kv_cache_config = kv
    runner.attn_groups = attn
    runner.num_ubatches = 2
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.ubatch_query_start_loc = [
        torch.zeros(MAX_REQS + 1, dtype=torch.int32) for _ in range(2)
    ]
    runner.ubatch_seq_lens = [
        torch.zeros(MAX_REQS, dtype=torch.int32) for _ in range(2)
    ]
    runner.ubatch_dcp_local_seq_lens = [None, None]
    runner._make_forward_contexts = lambda slices, metadata, slots, padding: [
        SimpleNamespace(attn_metadata=m) for m in metadata
    ]
    return (state, runner, kv, attn, forward)


def call_prepare(api, runner, inp, tables, full=False, capture=False):
    return api.prepare(
        runner,
        inp,
        tables,
        torch.zeros((len(tables), inp.num_tokens_after_padding), dtype=torch.int64),
        api.CUDAGraphMode.FULL if full else api.CUDAGraphMode.NONE,
        capture,
    )


def metadata(result, stage, group=0):
    return result.forward_contexts[stage].attn_metadata[f"gdn{group}.a"]


@pytest.mark.parametrize("prior", [0, 16])
def test_straddling_request_uses_full_step_ids_and_native_context(api, prior):
    state, runner, kv, attn, _ = setup(api, groups=2)
    inp = batch(api, [64], [prior + 64], slots=[7], has_prefill=True)
    tables = (
        torch.tensor([[3, 0, 0, 17, 21]], dtype=torch.int32),
        torch.tensor([[4, 0, 0, 27, 31]], dtype=torch.int32),
    )
    originals = [x.clone() for x in tables]
    result = call_prepare(api, runner, inp, tables)
    for stage in (0, 1):
        for gid, expected in enumerate([17, 27] if prior == 0 else [21, 31]):
            meta = metadata(result, stage, gid)
            assert meta.non_spec_state_indices_tensor.tolist() == [expected]
            assert meta.prefill_state_indices.tolist() == [expected]
            assert meta.prefill_has_initial_state.tolist() == [prior > 0 or stage == 1]
        assert runner.ubatch_query_start_loc[stage][:2].tolist() == [0, 32]
        assert runner.ubatch_seq_lens[stage][0].item() == prior + 32 * (stage + 1)
    for actual, original in zip(tables, originals, strict=True):
        torch.testing.assert_close(actual, original)
    assert state._full_batch_state_ids is None


def test_full_graph_stages_isolate_values_and_reuse_addresses(api):
    state, runner, _, attn, _ = setup(api, full=True)
    inp = batch(api, [1, 1, 1, 1], [16] * 4, slots=[7, 2, 8, 1])
    table = torch.tensor([[11], [12], [13], [14]], dtype=torch.int32)
    result = call_prepare(api, runner, inp, (table,), full=True)
    assert metadata(result, 0).non_spec_state_indices_tensor.tolist() == [11, 12]
    assert metadata(result, 1).non_spec_state_indices_tensor.tolist() == [13, 14]
    ptrs = [
        metadata(result, i).non_spec_state_indices_tensor.data_ptr() for i in (0, 1)
    ]
    assert ptrs[0] != ptrs[1]
    second = call_prepare(api, runner, inp, (table + 20,), full=True)
    assert [
        metadata(second, i).non_spec_state_indices_tensor.data_ptr() for i in (0, 1)
    ] == ptrs
    assert metadata(second, 0).non_spec_state_indices_tensor.tolist() == [31, 32]
    assert metadata(second, 1).non_spec_state_indices_tensor.tolist() == [33, 34]
    assert state._state_ids_by_req_slot[0][7].item() == 31
    assert state._state_ids_by_req_slot[0][0].item() == 0


@pytest.mark.parametrize("mode", ["align", "none"])
def test_full_padding_only_stage_has_null_ids(api, mode):
    _, runner, _, _, _ = setup(api, mode=mode, full=True)
    inp = batch(api, [1], [16], slots=[7], padded=8)
    table = torch.full((8, 1), 19, dtype=torch.int32)
    before = table.clone()
    result = call_prepare(api, runner, inp, (table,), full=True)
    first_ids = metadata(result, 0).non_spec_state_indices_tensor.tolist()
    assert first_ids[0] == 19 and all(i == 0 for i in first_ids[1:])
    assert set(metadata(result, 1).non_spec_state_indices_tensor.tolist()) == {0}
    torch.testing.assert_close(table, before)


def test_none_eager_retains_block_table_and_prefill_aliases(api):
    _, runner, _, _, _ = setup(api, mode="none")
    inp = batch(api, [7, 9], [27, 29])
    table = torch.tensor([[11], [12]], dtype=torch.int32)
    before = table.clone()
    result = call_prepare(api, runner, inp, (table,))
    torch.testing.assert_close(table, before)
    assert metadata(result, 0).non_spec_state_indices_tensor.tolist() == [11, 12]
    assert metadata(result, 1).non_spec_state_indices_tensor.tolist() == [12]
    assert (
        metadata(result, 1).prefill_state_indices.data_ptr()
        == metadata(result, 1).non_spec_state_indices_tensor.data_ptr()
    )


@pytest.mark.parametrize("mode", ["align", "none"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_fresh_seven_decode_nine_prefill_clears_only_selected_rows_once(
    api, mode, noncontiguous
):
    state, runner, _, _, forward = setup(
        api, mode=mode, groups=2, noncontiguous=noncontiguous
    )
    inp = batch(
        api,
        [1] * 7 + [9],
        [16] * 7 + [9],
        slots=[8, 9, 10, 11, 12, 13, 14, 3],
        has_prefill=True,
    )
    tables = tuple(
        torch.tensor([[i + 1 + gid * 10] for i in range(8)], dtype=torch.int32)
        for gid in range(2)
    )
    before = {
        name: tuple(t.clone() for t in layer.kv_cache)
        for name, layer in forward.items()
    }
    seen = []
    original = state.prepare_attn

    def recording(*args, **kwargs):
        stage = kwargs["ubatch_idx"]
        for gid in range(2):
            target = 8 + gid * 10
            for name in (f"gdn{gid}.a", f"gdn{gid}.b"):
                for actual, initial in zip(
                    forward[name].kv_cache, before[name], strict=True
                ):
                    if stage == 0:
                        assert torch.count_nonzero(actual[target]).item() == 0
                        actual[target].fill_(700)
                    else:
                        assert torch.all(actual[target] == 700)
                    keep = [i for i in range(64) if i not in (0, target)]
                    torch.testing.assert_close(actual[keep], initial[keep])
        seen.append(stage)
        return original(*args, **kwargs)

    state.prepare_attn = recording
    result = call_prepare(api, runner, inp, tables)
    assert seen == [0, 1]
    assert metadata(result, 0).num_decodes == 8
    assert metadata(result, 0).has_initial_state is None
    assert metadata(result, 1).prefill_has_initial_state.tolist() == [True]
    assert metadata(result, 1).prefill_state_indices.tolist() == [8]
    assert state._full_batch_state_ids is None
    state.prepare_attn = original
    continued = batch(api, [8], [17], slots=[3], has_prefill=True)
    continued_tables = tuple(
        torch.tensor([[8 + gid * 10, 8 + gid * 10]], dtype=torch.int32)
        for gid in range(2)
    )
    call_prepare(api, runner, continued, continued_tables)
    for gid in range(2):
        for tensor in forward[f"gdn{gid}.a"].kv_cache:
            assert torch.all(tensor[8 + gid * 10] == 700)
            assert torch.count_nonzero(tensor[0]).item() == 0


@pytest.mark.parametrize(
    "has_prefill,capture", [(False, False), (True, True), (False, True)]
)
def test_decode_dummy_and_capture_do_not_initialize_cache(api, has_prefill, capture):
    _, runner, _, _, forward = setup(api, full=capture)
    inp = batch(api, [1, 1], [1, 1], has_prefill=has_prefill)
    tables = (torch.tensor([[3], [4]], dtype=torch.int32),)
    before = {
        name: tuple(t.clone() for t in layer.kv_cache)
        for name, layer in forward.items()
    }
    call_prepare(api, runner, inp, tables, full=capture, capture=capture)
    for name, layer in forward.items():
        for actual, initial in zip(layer.kv_cache, before[name], strict=True):
            torch.testing.assert_close(actual, initial)


def test_scope_restored_when_native_prepare_raises(api):
    state, runner, _, _, _ = setup(api)
    inp = batch(api, [8], [8], has_prefill=True)

    def fail(*a, **k):
        raise ValueError("native builder failure")

    state.prepare_attn = fail
    with pytest.raises(ValueError, match="native builder failure"):
        call_prepare(api, runner, inp, (torch.tensor([[3]], dtype=torch.int32),))
    assert state._full_batch_state_ids is None


@pytest.fixture
def config_api(api, monkeypatch):
    from vllm.platforms import current_platform

    from afd_plugin.compat.patches import config_validation as patch

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    ns = dict(vars(patch))
    tree = ast.parse((UPSTREAM / "vllm/config/vllm.py").read_text())
    node = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "VllmConfig"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_dbo_unsupported_features"
    )
    node.name = "native_blockers"
    exec(
        compile(
            ast.Module(
                ast.parse("from __future__ import annotations").body + [node], []
            ),
            str(UPSTREAM / "vllm/config/vllm.py"),
            "exec",
        ),
        ns,
    )
    return SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


def config(api, role="attention"):
    return SimpleNamespace(
        additional_config={
            "afd": {"role": role, "num_attention_ranks": 2, "num_ffn_ranks": 1}
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"]
            ),
            multimodal_config=SimpleNamespace(
                language_model_only=True, enable_mm_embeds=False
            ),
            is_multimodal_model=True,
            is_hybrid=True,
            is_encoder_decoder=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=2 if role == "attention" else 1,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            num_ubatches=2,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False),
            cudagraph_mode=api.CUDAGraphMode.FULL,
        ),
        lora_config=None,
        speculative_config=None,
        is_mm_encoder_only=False,
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "architecture",
    ["Qwen3_5MoeForConditionalGeneration", "AFDQwen3_5MoeForConditionalGeneration"],
)
def test_actual_config_guards_accept_native_and_remapped_supported_lane(
    api, config_api, role, architecture
):
    cfg = config(api, role)
    cfg.model_config.hf_config.architectures = [architecture]
    original = copy.deepcopy(cfg)
    assert config_api.native_blockers(cfg) == [
        "dual batch overlap with multimodal models",
        "dual batch overlap with hybrid models",
    ]
    assert config_api._get_dbo_unsupported_features(cfg) == []
    assert cfg == original


@pytest.mark.parametrize(
    "mutation",
    [
        "non_afd",
        "other_hybrid",
        "multimodal",
        "mm_embeds",
        "topology",
        "ep",
        "elastic",
        "eplb",
        "sp",
        "num_ubatches",
        "gate",
    ],
)
def test_actual_config_guards_keep_unimplemented_models_and_modes(
    api, config_api, mutation
):
    cfg = config(api)
    if mutation == "non_afd":
        cfg.additional_config = {}
    if mutation == "other_hybrid":
        cfg.model_config.hf_config.architectures = ["OtherHybridModel"]
    if mutation == "multimodal":
        cfg.model_config.multimodal_config.language_model_only = False
    if mutation == "mm_embeds":
        cfg.model_config.multimodal_config.enable_mm_embeds = True
    if mutation == "topology":
        cfg.parallel_config.data_parallel_size = 1
    if mutation == "ep":
        cfg.parallel_config.enable_expert_parallel = False
    if mutation == "elastic":
        cfg.parallel_config.enable_elastic_ep = True
    if mutation == "eplb":
        cfg.parallel_config.enable_eplb = True
    if mutation == "sp":
        cfg.compilation_config.pass_config.enable_sp = True
    if mutation == "num_ubatches":
        cfg.parallel_config.num_ubatches = 3
    if mutation == "gate":
        cfg.additional_config["afd"]["compute_gate_on_attention"] = True
    assert config_api._get_dbo_unsupported_features(cfg) == config_api.native_blockers(
        cfg
    )


@pytest.mark.parametrize(
    "kind",
    ["lora", "spec", "pp", "pcp", "dcp", "piecewise", "encoder", "mm_encoder_only"],
)
def test_other_native_dbo_exclusions_remain(api, config_api, kind):
    cfg = config(api)
    if kind == "lora":
        cfg.lora_config = SimpleNamespace()
    if kind == "spec":
        cfg.speculative_config = SimpleNamespace()
    if kind == "pp":
        cfg.parallel_config.pipeline_parallel_size = 2
    if kind == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    if kind == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 2
    if kind == "piecewise":
        cfg.compilation_config.cudagraph_mode = api.CUDAGraphMode.PIECEWISE
    if kind == "encoder":
        cfg.model_config.is_encoder_decoder = True
    if kind == "mm_encoder_only":
        cfg.is_mm_encoder_only = True
    expected = [
        item
        for item in config_api.native_blockers(cfg)
        if item != "dual batch overlap with hybrid models"
        and (item != "dual batch overlap with multimodal models" or kind == "encoder")
    ]
    assert expected
    assert config_api._get_dbo_unsupported_features(cfg) == expected


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("qwen", [False, True])
def test_factory_binds_only_qwen_instance_at_all_dp_sizes(
    api, monkeypatch, dp_size, qwen
):

    class Runner:
        def __init__(self, *args):
            self.args = args

        def prepare(self, *args, **kwargs):
            return "native"

    original = Runner.prepare
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True, data_parallel_size=dp_size)
    )
    state = object.__new__(api.QwenGDNModelState) if qwen else object()
    from afd_plugin.compat.patches import gpu_mrv2_ubatching as patch

    monkeypatch.setattr(patch, "UBatchRunner", Runner)
    runner = api.maybe_build_ubatch_runner(
        cfg, "cpu", state, [], SimpleNamespace(), MAX_REQS
    )
    assert type(runner) is Runner
    assert ("prepare" in vars(runner)) is qwen
    assert ("_make_forward_contexts" in vars(runner)) is (dp_size == 1)
    assert Runner.prepare is original


def test_non_dbo_prepare_delegates_to_native_exactly(api):
    state, runner, kv, attn, _ = setup(api)
    inp = batch(api, [4], [20])
    table = torch.tensor([[6, 7]], dtype=torch.int32)
    output = state.prepare_attn(
        inp,
        api.CUDAGraphMode.NONE,
        (table,),
        torch.zeros((1, 4), dtype=torch.int64),
        attn,
        kv,
    )
    assert output["gdn0.a"].non_spec_state_indices_tensor.tolist() == [7]
    assert not state._state_ids_by_req_slot
    with pytest.raises(AssertionError, match="DBO is not supported"):
        state.prepare_attn(
            inp,
            api.CUDAGraphMode.NONE,
            (table,),
            torch.zeros((1, 4), dtype=torch.int64),
            attn,
            kv,
            ubatch_idx=1,
        )


def test_model_hook_selects_private_state_via_native_resolver(api, monkeypatch):
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls

    from afd_plugin.model_executor.models.qwen3_5 import (
        AFDQwen3_5MoeForConditionalGeneration,
    )

    model = object.__new__(AFDQwen3_5MoeForConditionalGeneration)
    assert resolve_model_state_cls(SimpleNamespace(), model) is api.QwenGDNModelState
    from vllm.config import VllmConfig

    from afd_plugin.compat.patches import config_validation

    assert (
        VllmConfig._get_dbo_unsupported_features
        is config_validation._get_dbo_unsupported_features
    )


def test_no_explicit_v2_keeps_native_opt_in_guard(api, config_api, monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    cfg = config(api)
    assert (
        config_api._get_dbo_unsupported_features(cfg)
        == config_api.native_blockers(cfg)
        == ["dual batch overlap"]
    )


def test_npu_does_not_use_gpu_config_exception(api, config_api, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    cfg = config(api)
    assert config_api._get_dbo_unsupported_features(cfg) == config_api.native_blockers(
        cfg
    )


def test_none_eager_padding_keeps_borrowed_table_and_zero_work(api):
    _, runner, _, _, _ = setup(api, mode="none")
    inp = batch(api, [1], [16], slots=[7], padded=8)
    table = torch.full((8, 1), 19, dtype=torch.int32)
    before = table.clone()
    result = call_prepare(api, runner, inp, (table,))
    torch.testing.assert_close(table, before)
    assert metadata(result, 1).num_actual_tokens == 0
    assert set(metadata(result, 1).non_spec_state_indices_tensor.tolist()) == {19}
