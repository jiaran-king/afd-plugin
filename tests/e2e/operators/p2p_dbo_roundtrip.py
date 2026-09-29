# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Two-GPU regression for the real DeepSeek proxy / NCCL DBO exchange.

Run through an allocation, with NCCL_P2P_DISABLE=1 to exercise SHM:
  torchrun --standalone --nproc-per-node=2 -m tests.e2e.operators.p2p_dbo_roundtrip

No model weights are needed. Rank 0 emulates layer/stage-ordered FFN work;
rank 1 runs the actual proxy with native vLLM ubatch contexts. The 8 MiB
payload is the issue398 profiling shape, not a tiny buffered-send probe.
"""

import os
import threading
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from vllm import forward_context
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.forward_context import ForwardContext
from vllm.v1.worker.ubatching import make_ubatch_contexts

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDForwardContextMetadata
from afd_plugin.connectors.gpu.p2p import (
    P2pNcclAFDConnector,
    _register_comm,
    _register_p2p_custom_ops,
    _TensorMetadata,
)
from afd_plugin.model_executor.models.deepseek_v2 import AFDAttentionFusedMoE

TOKENS = 2048
HIDDEN_SIZE = 2048
NUM_STAGES = 2
NUM_LAYERS = 3
ROUTER_SIZE = 64
JOIN_TIMEOUT_SECONDS = 60


def attention_roundtrip(connector: P2pNcclAFDConnector, use_dbo: bool) -> None:
    connector.vllm_config.parallel_config.enable_dbo = use_dbo
    device = torch.cuda.current_device()
    tensors = [
        torch.full(
            (TOKENS, HIDDEN_SIZE), stage + 1, device=device, dtype=torch.bfloat16
        )
        for stage in range(NUM_STAGES)
    ]
    routers = [
        torch.full(
            (TOKENS, ROUTER_SIZE), stage + 2, device=device, dtype=torch.bfloat16
        )
        for stage in range(NUM_STAGES)
    ]
    identity = torch.eye(HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    contexts = [
        ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={
                "afd_metadata": AFDForwardContextMetadata(
                    tokens_start_loc=[0, TOKENS],
                    requests_start_loc=[0, 1],
                    stage_idx=stage,
                    connector=connector,
                    tokens_lens=[TOKENS],
                    num_stages=NUM_STAGES,
                )
            },
        )
        for stage in range(NUM_STAGES)
    ]
    proxies = [
        AFDAttentionFusedMoE(layer_idx=layer, is_internal_router=False)
        for layer in range(NUM_LAYERS)
    ]
    errors = []

    def step(stage: int, layer: int) -> None:
        # A real producer on the compute stream exercises the comm dependency.
        tensors[stage] = tensors[stage] @ identity
        tensors[stage] = proxies[layer](tensors[stage], routers[stage])
        # A consumer immediately following the exchange detects missing waits.
        tensors[stage].add_(1)

    if use_dbo:
        barrier = threading.Barrier(NUM_STAGES + 1)
        compute_stream = torch.cuda.current_stream()
        comm_stream = torch.cuda.Stream()
        ubatches = make_ubatch_contexts(
            NUM_STAGES, compute_stream, comm_stream, contexts, barrier
        )

        def run_stage(stage: int) -> None:
            try:
                torch.cuda.set_device(device)
                with torch.inference_mode(), ubatches[stage]:
                    for layer in range(NUM_LAYERS):
                        step(stage, layer)
            except BaseException as error:
                errors.append(error)

        workers = [
            threading.Thread(target=run_stage, args=(stage,), daemon=True)
            for stage in range(NUM_STAGES)
        ]
        for worker in workers:
            worker.start()
        barrier.wait(timeout=JOIN_TIMEOUT_SECONDS)
        ubatches[0].cpu_wait_event.set()
        for worker in workers:
            worker.join(timeout=JOIN_TIMEOUT_SECONDS)
        if errors:
            raise errors[0]
        assert all(not worker.is_alive() for worker in workers), (
            "DBO CPU threads stalled"
        )
    else:
        with torch.inference_mode():
            for layer in range(NUM_LAYERS):
                for stage in range(NUM_STAGES):
                    forward_context._forward_context = contexts[stage]
                    step(stage, layer)

    torch.cuda.synchronize()
    for stage, tensor in enumerate(tensors):
        expected = stage + 1
        for layer in range(NUM_LAYERS):
            expected = expected * 2 + (stage + 2) + layer + 1
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, expected), rtol=0, atol=0
        )
    print(
        f"PASS: DBO={use_dbo}, two distinct stages, router payload, producer/consumer",
        flush=True,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=JOIN_TIMEOUT_SECONDS))
    assert dist.get_world_size() == 2
    group = StatelessProcessGroup.create(
        host=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]) + 1,
        rank=rank,
        world_size=2,
    )
    a2e = PyNcclCommunicator(group, device=local_rank)
    e2a = PyNcclCommunicator(group, device=local_rank)
    a2e_id, e2a_id = _register_comm(a2e), _register_comm(e2a)
    _register_p2p_custom_ops()
    # Configure only the transport under test; control-plane startup is covered
    # by the model E2E. All sends, receives and ubatch events below are real GPU work.
    config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS
            )
        ),
        parallel_config=SimpleNamespace(enable_dbo=False),
    )
    connector = P2pNcclAFDConnector(
        rank, local_rank, config, AFDConfig(role="attention"), 0
    )
    connector.a2e_group = connector.e2a_group = group
    connector.a2e_comm_id, connector.e2a_comm_id = a2e_id, e2a_id
    connector.tensor_metadata_list = {
        stage: _TensorMetadata(
            torch.device("cuda", local_rank),
            torch.bfloat16,
            torch.Size((TOKENS, HIDDEN_SIZE)),
        )
        for stage in range(NUM_STAGES)
    }
    received = torch.empty(
        (TOKENS, HIDDEN_SIZE), device=local_rank, dtype=torch.bfloat16
    )
    router = torch.empty((TOKENS, ROUTER_SIZE), device=local_rank, dtype=torch.bfloat16)
    trace_dir = os.environ.get("ISSUE398_TRACE_DIR")
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if trace_dir
        else nullcontext()
    )
    with profiler:
        for use_dbo in (False, True):
            dist.barrier()
            if rank == 1:
                attention_roundtrip(connector, use_dbo)
            else:
                for layer in range(NUM_LAYERS):
                    for _stage in range(NUM_STAGES):
                        a2e.recv(received, 1)
                        a2e.recv(router, 1)
                        received.mul_(2).add_(router[:, :1]).add_(layer)
                        e2a.send(received, 1)
                torch.cuda.synchronize()
            dist.barrier()
    if trace_dir:
        profiler.export_chrome_trace(
            str(Path(trace_dir) / f"roundtrip-rank-{rank}.json")
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
