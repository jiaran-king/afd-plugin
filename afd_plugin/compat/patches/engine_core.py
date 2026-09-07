# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""EngineCore compatibility patch for AFD FFN daemon mode.

The AFD FFN side runs as a connector daemon, not as a normal request-scheduling
EngineCore. After constructing the model executor, FFN EngineCore
initialization returns before KV cache and scheduler setup. This keeps FFN
startup out of HybridKVCacheCoordinator.
"""

from __future__ import annotations

import gc
import queue
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import vllm.v1.engine.core as core_module

from afd_plugin.config import AFDConfig, parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor
    from vllm.v1.kv_cache_interface import KVCacheConfig


_ORIGINAL_ENGINE_CORE_INIT_ATTR = "_afd_plugin_original_engine_core_init"
_ORIGINAL_ENGINE_CORE_KV_ATTR = "_afd_plugin_original_engine_core_initialize_kv_caches"
_ORIGINAL_ENGINE_CORE_SHUTDOWN_ATTR = "_afd_plugin_original_engine_core_shutdown"
_ORIGINAL_ENGINE_CORE_BUSY_LOOP_ATTR = "_afd_plugin_original_engine_core_busy_loop"
_ORIGINAL_DP_ENGINE_CORE_BUSY_LOOP_ATTR = (
    "_afd_plugin_original_dp_engine_core_busy_loop"
)

if not hasattr(core_module, _ORIGINAL_ENGINE_CORE_INIT_ATTR):
    setattr(
        core_module, _ORIGINAL_ENGINE_CORE_INIT_ATTR, core_module.EngineCore.__init__
    )
if not hasattr(core_module, _ORIGINAL_ENGINE_CORE_KV_ATTR):
    setattr(
        core_module,
        _ORIGINAL_ENGINE_CORE_KV_ATTR,
        core_module.EngineCore._initialize_kv_caches,
    )
if not hasattr(core_module, _ORIGINAL_ENGINE_CORE_SHUTDOWN_ATTR):
    setattr(
        core_module,
        _ORIGINAL_ENGINE_CORE_SHUTDOWN_ATTR,
        core_module.EngineCore.shutdown,
    )
if not hasattr(core_module, _ORIGINAL_ENGINE_CORE_BUSY_LOOP_ATTR):
    setattr(
        core_module,
        _ORIGINAL_ENGINE_CORE_BUSY_LOOP_ATTR,
        core_module.EngineCoreProc.run_busy_loop,
    )
if not hasattr(core_module, _ORIGINAL_DP_ENGINE_CORE_BUSY_LOOP_ATTR):
    setattr(
        core_module,
        _ORIGINAL_DP_ENGINE_CORE_BUSY_LOOP_ATTR,
        core_module.DPEngineCoreProc.run_busy_loop,
    )

_original_engine_core_init = getattr(core_module, _ORIGINAL_ENGINE_CORE_INIT_ATTR)
_original_engine_core_kv = getattr(core_module, _ORIGINAL_ENGINE_CORE_KV_ATTR)
_original_engine_core_shutdown = getattr(
    core_module, _ORIGINAL_ENGINE_CORE_SHUTDOWN_ATTR
)
_original_engine_core_busy_loop = getattr(
    core_module, _ORIGINAL_ENGINE_CORE_BUSY_LOOP_ATTR
)
_original_dp_engine_core_busy_loop = getattr(
    core_module, _ORIGINAL_DP_ENGINE_CORE_BUSY_LOOP_ATTR
)


# Patch reason: AFD FFN ranks run as connector daemons instead of normal
# request-scheduling EngineCore instances.
# Patch functionality: returns after model executor construction for AFD FFN
# configs while preserving the target upstream tag's normal EngineCore startup
# logic for non-AFD configs.
# Signature: matches upstream; no added parameters.
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None = None,
    include_finished_set: bool = False,
) -> None:
    # ### PATCH START: AFD FFN EngineCore daemon initialization
    # FFN ranks are connector daemons, so stop EngineCore initialization after
    # executor construction instead of setting up KV cache and scheduler state.
    if _is_afd_ffn_config(vllm_config):
        _initialize_ffn_engine_core(
            self,
            core_module,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
        )
        return
    # ### PATCH END: AFD FFN EngineCore daemon initialization

    # Delegation exception: vLLM v0.28's normal constructor gained new state
    # (including weight-version and EC-output ownership). Reuse the exact
    # target implementation for every non-AFD engine instead of retaining a
    # second copied constructor that would drift on the next upstream change.
    return _original_engine_core_init(
        self,
        vllm_config,
        executor_class,
        log_stats,
        executor_fail_callback,
        include_finished_set,
    )


# Patch reason: AFD FFN daemon engines skip scheduler/KV setup, so upstream
# shutdown can touch attributes that were intentionally not initialized.
# Patch functionality: stops the connector worker loop and shuts down the model
# executor for AFD FFN engines while preserving upstream shutdown for non-AFD
# engines.
# Signature: matches upstream; no added parameters.
def shutdown(self) -> None:
    # ### PATCH START: AFD FFN EngineCore shutdown
    # Stop the connector-driven worker loop before shutting down the executor;
    # scheduler/KV state may not exist for FFN daemon engines.
    if _is_afd_ffn_engine(self):
        _stop_ffn_worker_loop(self)
        model_executor = getattr(self, "model_executor", None)
        if model_executor is not None:
            model_executor.shutdown()
        with suppress(Exception):
            gc.unfreeze()
        core_module.cleanup_dist_env_and_memory()
        return
    # ### PATCH END: AFD FFN EngineCore shutdown

    # Delegation exception: target shutdown now also tears down its additional
    # engine state. Only the AFD FFN path above needs custom cleanup.
    return _original_engine_core_shutdown(self)


# Patch reason: late-loaded AFD FFN EngineCore paths may ask for KV cache setup
# even though FFN daemons do not own request KV blocks.
# Patch functionality: returns a minimal KV-cache-shaped result for AFD FFN
# configs while preserving upstream KV cache initialization for non-AFD configs.
# Signature: matches upstream; no added parameters.
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
    # ### PATCH START: AFD FFN late-loaded KV cache bypass
    # FFN daemon engines do not schedule requests or own KV cache blocks, but
    # late-loaded paths still need a minimal KV cache config-shaped result.
    if _is_afd_ffn_config(vllm_config):
        _prepare_late_loaded_ffn_engine_core(self, vllm_config)
        return _AFDFFNKVCacheConfig()
    # ### PATCH END: AFD FFN late-loaded KV cache bypass

    # Delegation exception: the target v0.28 implementation changed KV-cache
    # capacity updates and warmup ordering. Preserve that exact implementation
    # for non-FFN engines; this patch owns only the AFD daemon bypass above.
    return _original_engine_core_kv(self, vllm_config)


# Patch reason: AFD FFN ranks must run the connector server loop rather than
# vLLM's normal request scheduling busy loop.
# Patch functionality: starts and monitors the FFN connector loop for AFD FFN
# engines while preserving upstream busy loops for non-AFD engine processes.
# Signature: matches upstream; no added parameters.
def run_busy_loop(self) -> None:
    # ### PATCH START: AFD FFN connector busy loop
    # FFN ranks run the connector server loop and poll worker-side failures
    # instead of executing vLLM's normal request scheduling loop.
    if _is_afd_ffn_engine(self):
        result = _run_ffn_busy_loop(self, core_module)
        return result
    # ### PATCH END: AFD FFN connector busy loop

    # Delegation exception: vLLM v0.28 added fault-tolerant wrappers, request
    # count publication, and new elastic-EP state transitions to both native
    # busy loops. Preserve those target implementations for non-FFN engines.
    if isinstance(self, core_module.DPEngineCoreProc):
        return _original_dp_engine_core_busy_loop(self)
    return _original_engine_core_busy_loop(self)


class _AFDFFNKVCacheConfig:
    kv_cache_groups: list[Any] = []


class _AFDFFNNoopScheduler:
    connector = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_kv_connector(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def has_requests(self) -> bool:
        return False

    def has_unfinished_requests(self) -> bool:
        return False

    def get_num_unfinished_requests(self) -> int:
        return 0

    def finish_requests(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _initialize_ffn_engine_core(
    self,
    core_module: Any,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None,
) -> None:
    try:
        from vllm.plugins import load_general_plugins

        load_general_plugins()
    except Exception:
        core_module.logger.debug(
            "AFD FFN EngineCore could not reload vLLM plugins",
            exc_info=True,
        )

    afd_config = _get_afd_config(vllm_config)
    self.vllm_config = vllm_config
    self.afd_config = afd_config
    self.log_stats = log_stats

    with suppress(Exception):
        vllm_config.afd_config = afd_config

    parallel_config = getattr(vllm_config, "parallel_config", None)
    local_dp_rank = getattr(parallel_config, "data_parallel_rank_local", 0)
    if not local_dp_rank:
        version = getattr(core_module, "VLLM_VERSION", "unknown")
        core_module.logger.info(
            "Initializing an AFD FFN V1 engine (v%s) with config: %s",
            version,
            vllm_config,
        )

    self.model_executor = executor_class(vllm_config)
    if executor_fail_callback is not None:
        self.model_executor.register_failure_callback(executor_fail_callback)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None:
        cache_config.num_gpu_blocks = 0
        cache_config.num_cpu_blocks = 0

    # These attributes let common shutdown/debug utility paths tolerate the
    # intentionally skipped KV/scheduler initialization.
    self.available_gpu_memory_for_kv_cache = -1
    self.structured_output_manager = None
    self.scheduler = None
    self.mm_receiver_cache = None
    self.batch_queue_size = 0
    self.batch_queue = None
    self.request_block_hasher = None
    self.aborts_queue = queue.Queue()
    self._idle_state_callbacks = []
    self.use_spec_decode = False
    self.is_pooling_model = False
    self.is_ec_consumer = True


def _prepare_late_loaded_ffn_engine_core(
    self,
    vllm_config: VllmConfig,
) -> None:
    afd_config = _get_afd_config(vllm_config)
    self.afd_config = afd_config
    with suppress(Exception):
        vllm_config.afd_config = afd_config

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None:
        cache_config.num_gpu_blocks = 0
        cache_config.num_cpu_blocks = 0
        with suppress(Exception):
            cache_config.enable_prefix_caching = False

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is not None:
        with suppress(Exception):
            scheduler_config.enable_chunked_prefill = False
        with suppress(Exception):
            scheduler_config.get_scheduler_cls = lambda: _AFDFFNNoopScheduler


def _run_ffn_busy_loop(self, core_module: Any) -> None:
    core_module.logger.info("AFD FFN EngineCore started; workers run connector loop.")

    started = False
    try:
        self.model_executor.collective_rpc("start_ffn_server_loop")
        started = True
        while _is_running(self, core_module):
            self.model_executor.collective_rpc("raise_ffn_loop_error_if_any")
            time.sleep(0.5)
    except KeyboardInterrupt:
        core_module.logger.info(
            "AFD FFN EngineCore shutting down after KeyboardInterrupt"
        )
    except Exception:
        core_module.logger.exception("AFD FFN EngineCore encountered a fatal error")
        raise
    finally:
        if started:
            _stop_ffn_worker_loop(self)

    raise SystemExit


def _stop_ffn_worker_loop(self) -> None:
    model_executor = getattr(self, "model_executor", None)
    if model_executor is None:
        return
    try:
        model_executor.collective_rpc("stop_ffn_server_loop")
    except Exception:
        core_module.logger.debug(
            "AFD FFN worker loop stop failed or was already stopped",
            exc_info=True,
        )


def _is_running(self, core_module: Any) -> bool:
    shutdown_state = getattr(self, "shutdown_state", None)
    engine_shutdown_state = getattr(core_module, "EngineShutdownState", None)
    running_state = getattr(engine_shutdown_state, "RUNNING", None)
    if shutdown_state is None or running_state is None:
        return True
    return shutdown_state == running_state


def _is_afd_ffn_engine(self) -> bool:
    return _is_afd_ffn_config(getattr(self, "vllm_config", None))


def _is_afd_ffn_config(vllm_config: VllmConfig | None) -> bool:
    config = _get_afd_config(vllm_config)
    return config is not None and config.role == "ffn"


def _get_afd_config(vllm_config: VllmConfig | None) -> AFDConfig | None:
    existing = getattr(vllm_config, "afd_config", None)
    if isinstance(existing, AFDConfig):
        return existing
    try:
        return parse_optional_afd_config(vllm_config, validate=False)
    except Exception:
        core_module.logger.debug(
            "Unable to parse AFD config from vLLM config",
            exc_info=True,
        )
        return None


core_module.EngineCore.__init__ = __init__
core_module.EngineCore._initialize_kv_caches = _initialize_kv_caches
core_module.EngineCore.shutdown = shutdown
core_module.EngineCoreProc.run_busy_loop = run_busy_loop
core_module.DPEngineCoreProc.run_busy_loop = run_busy_loop
core_module.logger.debug("AFD EngineCore patch applied")


__all__: list[str] = []
