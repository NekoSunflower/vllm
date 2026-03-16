# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import gc
import sys
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    # GDN (Gated Delta Net) Triton kernel warmup.
    # Re-compiles only the winning configs (autotuner cache hit from
    # the earlier gdn_warmup_and_cleanup call).
    _gdn_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def _gdn_warmup(worker: "Worker"):
    """Warm up GDN (Gated Delta Net) Triton kernels.

    The Triton autotuner cache is process-global, so warming up a single
    GDN layer is sufficient — all layers share the same kernel configs.
    """
    model = worker.get_model()
    for module in model.modules():
        if hasattr(module, "_warmup_prefill_kernels"):
            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype
            dummy = torch.empty(1, device=device, dtype=dtype)
            module._warmup_prefill_kernels(dummy)
            break


def _unload_fla_triton_modules() -> int:
    """Unload CUDA modules from Triton JIT caches for FLA ops.

    After Triton autotuning, the JIT cache holds ``CompiledKernel``
    objects for ALL benchmarked configs (~300 across FLA ops). Only the
    winning config per autotune key is used going forward, but Triton
    never calls ``cuModuleUnload`` — the modules persist for the
    lifetime of the process.

    This function walks every Triton ``Autotuner`` instance in the
    imported FLA ops modules, accesses the wrapped ``JITFunction``'s
    ``device_caches``, and calls ``cuModuleUnload`` on each compiled
    kernel's CUDA module handle.  The ``Autotuner.cache`` (winning
    ``Config`` per key) is **not** touched, so subsequent kernel calls
    skip benchmarking and compile only the winning config.

    Returns the number of CUDA modules successfully unloaded.
    """
    try:
        import ctypes

        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except ImportError:
        return 0

    # Load the CUDA driver library for cuModuleUnload.
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        logger.debug("Could not load libcuda.so.1; skipping module unload.")
        return 0

    cu_module_unload = libcuda.cuModuleUnload
    cu_module_unload.argtypes = [ctypes.c_void_p]
    cu_module_unload.restype = ctypes.c_int  # CUresult; 0 = CUDA_SUCCESS

    # Ensure all GPU work is complete before unloading modules.
    current_platform.synchronize()

    # Discover Autotuner instances from imported FLA ops modules.
    # FLA kernels use two decorator patterns:
    #   Heuristics(Autotuner(JITFunction))  — most kernels
    #   Autotuner(JITFunction)              — e.g. l2norm
    fla_prefix = "vllm.model_executor.layers.fla.ops"
    autotuners: list[tuple[str, JITFunction]] = []

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith(fla_prefix):
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if obj is None:
                continue
            # Walk the .fn chain to find Autotuner → JITFunction.
            fn = obj
            while fn is not None and not isinstance(fn, Autotuner):
                fn = getattr(fn, "fn", None)
            if not isinstance(fn, Autotuner):
                continue
            # fn is the Autotuner; fn.fn should be the JITFunction.
            jit_fn = fn.fn
            while jit_fn is not None and not isinstance(jit_fn, JITFunction):
                jit_fn = getattr(jit_fn, "fn", None)
            if isinstance(jit_fn, JITFunction):
                autotuners.append((attr_name, jit_fn))

    n_unloaded = 0
    for name, jit_fn in autotuners:
        if not hasattr(jit_fn, "device_caches"):
            continue
        for device_key in list(jit_fn.device_caches.keys()):
            cache_tuple = jit_fn.device_caches[device_key]
            if not isinstance(cache_tuple, (tuple, list)) or len(cache_tuple) < 2:
                continue
            kernel_cache = cache_tuple[0]
            if not isinstance(kernel_cache, dict):
                continue
            for compiled_kernel in kernel_cache.values():
                module_handle = getattr(compiled_kernel, "module", None)
                if module_handle is not None and isinstance(module_handle, int):
                    result = cu_module_unload(ctypes.c_void_p(module_handle))
                    if result == 0:  # CUDA_SUCCESS
                        n_unloaded += 1
            kernel_cache.clear()
            # Also clear the kernel_key_cache (second element).
            kernel_key_cache = cache_tuple[1]
            if isinstance(kernel_key_cache, dict):
                kernel_key_cache.clear()

    return n_unloaded


def gdn_warmup_and_cleanup(worker: "Worker") -> None:
    """Pre-profiling GDN warmup with CUDA module cleanup.

    Runs GDN kernel warmup to populate Triton autotuner caches with
    winning configs, then unloads all compiled CUDA modules to free
    GPU memory (~1.5 GiB).  This ensures:

    1. Autotuner results are cached (no re-benchmarking on next call).
    2. CUDA module memory does not inflate ``non_torch_increase``
       during memory profiling.
    3. KV cache gets the maximum available memory budget.

    On subsequent GDN kernel calls (e.g. in ``kernel_warmup()`` after
    KV cache allocation), the autotuner finds cached winning configs
    and compiles only those — loading ~1 CUDA module per kernel instead
    of ~20.
    """
    model = worker.get_model()
    if not any(hasattr(m, "_warmup_prefill_kernels") for m in model.modules()):
        return

    logger.info("Running GDN Triton kernel warmup before memory profiling.")

    # FLA Triton kernels require a PyTorch-backed allocator for scratch
    # memory.  Set it before warmup so autotuning can run.
    from vllm.triton_utils.allocation import set_triton_allocator

    set_triton_allocator(worker.device)
    _gdn_warmup(worker)

    try:
        n_unloaded = _unload_fla_triton_modules()
        if n_unloaded > 0:
            logger.info(
                "Unloaded %d Triton CUDA modules to free GPU memory for KV cache.",
                n_unloaded,
            )
    except Exception:
        logger.warning(
            "Failed to unload Triton CUDA modules. Memory profiling "
            "may over-estimate non-KV-cache memory usage.",
            exc_info=True,
        )

    gc.collect()
    torch.accelerator.empty_cache()


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    import vllm.utils.flashinfer as fi_utils

    with torch.inference_mode(), fi_utils.autotune():
        # Certain FlashInfer kernels (e.g. nvfp4 routed moe) are
        # incompatible with autotuning. This state is used to skip
        # those kernels during the autotuning process.
        fi_utils._is_fi_autotuning = True

        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )

        fi_utils._is_fi_autotuning = False
