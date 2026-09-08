# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import ast
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

# Exact constructor contract from vllm/v1/worker/gpu_ubatch_wrapper.py at
# 6b5a12c0f843f10aec5fd5e439a1b8091dd66b8c. Keep the module-level factory call:
# the old AFD super().__init__ implementation must exercise this new contract.
NATIVE_CONSTRUCTOR = """
def __init__(
    self,
    runnable: Callable,
    vllm_config: VllmConfig,
    runtime_mode: CUDAGraphMode,
    device: torch.cuda.device,
):
    self.runnable = runnable
    self.vllm_config = vllm_config
    self.compilation_config = vllm_config.compilation_config
    self.comm_stream = torch.cuda.Stream(device=device)
    # Ubatch threads plus the main thread
    self.ready_barrier = threading.Barrier(
        self.vllm_config.parallel_config.num_ubatches + 1
    )

    self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

    self.cudagraph_wrapper = None
    if runtime_mode is not CUDAGraphMode.NONE:
        self.cudagraph_wrapper = CUDAGraphWrapper(
            runnable, vllm_config, runtime_mode=runtime_mode
        )

    self.sm_control = create_sm_control_context(vllm_config.parallel_config)
    self.device = device
    self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
    self._runnable_str = str(runnable) if self.is_debugging_mode else None
"""


@pytest.mark.parametrize("afd_active", [True, False])
@pytest.mark.parametrize("use_graph", [True, False])
@pytest.mark.parametrize("debug", [True, False])
def test_constructor_preserves_native_fields_and_selects_sm_context(
    afd_active, use_graph, debug
):
    source_path = (
        Path(__file__).resolve().parents[4] / "afd_plugin/v1/worker/ubatch_wrapper.py"
    )
    module = ast.parse(source_path.read_text())
    wrapper = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDUBatchWrapper"
    )
    # Isolate only construction, retaining the old override if present so this
    # test also reproduces its bypass by the new native constructor.
    wrapper.body = [
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"__init__", "_create_sm_control_context"}
    ]
    modes = SimpleNamespace(NONE=sentinel.no_graph, FULL=sentinel.full_graph)
    runtime_mode = modes.FULL if use_graph else modes.NONE
    config = SimpleNamespace(
        compilation_config=sentinel.compilation_config,
        parallel_config=SimpleNamespace(num_ubatches=2),
    )
    stream = Mock(return_value=sentinel.comm_stream)
    graph_wrapper = Mock(return_value=sentinel.graph_wrapper)
    native_factory = Mock(return_value=sentinel.native_context)
    if afd_active:
        native_factory.side_effect = AssertionError("AFD invoked native SM factory")
    active_check = Mock(return_value=afd_active)
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(Stream=stream)),
        "threading": threading,
        "CUDAGraphMode": modes,
        "CUDAGraphWrapper": graph_wrapper,
        "create_sm_control_context": native_factory,
        "is_afd_active": active_check,
        "nullcontext": nullcontext,
        "envs": SimpleNamespace(VLLM_LOGGING_LEVEL="DEBUG" if debug else "INFO"),
    }
    native = ast.ClassDef(
        name="UBatchWrapper",
        bases=[],
        keywords=[],
        body=ast.parse(NATIVE_CONSTRUCTOR).body,
        decorator_list=[],
    )
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            native,
            wrapper,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(isolated), str(source_path), "exec"),
        namespace,
    )
    runnable = Mock(name="runnable")
    instance = namespace["AFDUBatchWrapper"](
        runnable, config, runtime_mode, sentinel.device
    )

    stream.assert_called_once_with(device=sentinel.device)
    assert instance.ready_barrier.parties == config.parallel_config.num_ubatches + 1
    assert instance.ready_barrier.n_waiting == 0
    if use_graph:
        graph_wrapper.assert_called_once_with(
            runnable, config, runtime_mode=runtime_mode
        )
    else:
        graph_wrapper.assert_not_called()
    active_check.assert_called_once_with(config)
    if afd_active:
        native_factory.assert_not_called()
        assert isinstance(instance.sm_control, nullcontext)
        with instance.sm_control as context:
            assert context is None
    else:
        native_factory.assert_called_once_with(config.parallel_config)
        assert native_factory.call_args.args[0] is config.parallel_config
        assert instance.sm_control is sentinel.native_context

    # Exact field set catches missing and unexpected native initialization.
    assert vars(instance) == {
        "runnable": runnable,
        "vllm_config": config,
        "compilation_config": sentinel.compilation_config,
        "comm_stream": sentinel.comm_stream,
        "ready_barrier": instance.ready_barrier,
        "cudagraphs": {},
        "cudagraph_wrapper": sentinel.graph_wrapper if use_graph else None,
        "sm_control": instance.sm_control,
        "device": sentinel.device,
        "is_debugging_mode": debug,
        "_runnable_str": str(runnable) if debug else None,
        "_afd_context_provider": None,
    }
