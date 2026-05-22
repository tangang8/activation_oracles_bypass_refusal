from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from distributed_utils import DistributedContext


def _bytes_to_gb(n: float) -> float:
    return float(n) / (1024.0**3)


@dataclass
class _NvmlSample:
    gpu_util_percent: float
    memory_util_percent: float
    power_watts: float | None


class _NvmlSampler:
    def __init__(self, device_index: int, interval_s: float = 0.1):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: list[_NvmlSample] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active = False

        self._pynvml = None
        self._handle = None
        try:
            import pynvml  # type: ignore

            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._active = True
        except Exception:
            self._active = False

    def start(self) -> None:
        if not self._active:
            return

        def run() -> None:
            assert self._pynvml is not None
            assert self._handle is not None
            while not self._stop.is_set():
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    try:
                        power_mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                        power_w = float(power_mw) / 1000.0
                    except Exception:
                        power_w = None
                    self.samples.append(
                        _NvmlSample(
                            gpu_util_percent=float(util.gpu),
                            memory_util_percent=float(util.memory),
                            power_watts=power_w,
                        )
                    )
                except Exception:
                    break
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        if not self._active:
            return {}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {}

        gpu_vals = [s.gpu_util_percent for s in self.samples]
        mem_vals = [s.memory_util_percent for s in self.samples]
        power_vals = [s.power_watts for s in self.samples if s.power_watts is not None]
        out = {
            "gpu/avg_utilization_percent": sum(gpu_vals) / len(gpu_vals),
            "gpu/max_utilization_percent": max(gpu_vals),
            "gpu/avg_memory_utilization_percent": sum(mem_vals) / len(mem_vals),
        }
        if power_vals:
            out["gpu/avg_power_watts"] = sum(power_vals) / len(power_vals)
        return out


class PerfLogger:
    def __init__(
        self,
        wandb_run: Any | None,
        dist_ctx: DistributedContext,
        enabled: bool = True,
        include_non_main: bool = False,
        gpu_sampling: bool = False,
    ):
        self.wandb_run = wandb_run
        self.dist_ctx = dist_ctx
        self.enabled = enabled
        self.include_non_main = include_non_main
        self.gpu_sampling = gpu_sampling
        self.step = 0
        self.summary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    @property
    def can_log(self) -> bool:
        if not self.enabled or self.wandb_run is None:
            return False
        if self.include_non_main:
            return True
        return self.dist_ctx.is_main

    def log_event(self, name: str, metrics: dict[str, float], metadata: dict[str, Any] | None = None) -> None:
        if not self.can_log:
            return
        payload: dict[str, Any] = {"perf/event_name": name, "perf/step": self.step}
        payload.update(metrics)
        if metadata:
            for k, v in metadata.items():
                payload[f"meta/{k}"] = v
        # Let W&B manage the global step so perf logs don't collide with other
        # module logs that don't provide an explicit step.
        self.wandb_run.log(payload)
        self.step += 1

    @contextmanager
    def track(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[dict[str, float]]:
        metrics: dict[str, float] = {}
        if not self.enabled:
            yield metrics
            return

        has_cuda = torch.cuda.is_available() and self.dist_ctx.device.type == "cuda"
        device = self.dist_ctx.device
        nvml = None
        had_cuda_error = False
        if has_cuda:
            try:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            except RuntimeError:
                had_cuda_error = True
            if self.gpu_sampling:
                index = device.index if device.index is not None else 0
                nvml = _NvmlSampler(index)
                nvml.start()

        t0 = time.perf_counter()
        try:
            yield metrics
        finally:
            elapsed = time.perf_counter() - t0
            metrics["perf/elapsed_seconds"] = elapsed

            if has_cuda and not had_cuda_error:
                try:
                    torch.cuda.synchronize(device)
                    metrics.update(
                        {
                            "gpu/memory_allocated_gb": _bytes_to_gb(torch.cuda.memory_allocated(device)),
                            "gpu/memory_reserved_gb": _bytes_to_gb(torch.cuda.memory_reserved(device)),
                            "gpu/peak_memory_allocated_gb": _bytes_to_gb(torch.cuda.max_memory_allocated(device)),
                            "gpu/peak_memory_reserved_gb": _bytes_to_gb(torch.cuda.max_memory_reserved(device)),
                        }
                    )
                except RuntimeError:
                    had_cuda_error = True
                    metrics["perf/cuda_metrics_unavailable"] = 1.0
                finally:
                    if nvml is not None:
                        metrics.update(nvml.stop())
            elif nvml is not None:
                metrics.update(nvml.stop())

            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)):
                    self.summary[name][mk] += float(mv)
            self.summary[name]["_count"] += 1.0

            self.log_event(name, metrics, metadata=metadata)

    def flush_summary(self) -> None:
        if not self.can_log:
            return
        for event_name, vals in self.summary.items():
            count = vals.get("_count", 0.0)
            if count <= 0:
                continue
            payload: dict[str, float] = {"perf/summary_count": count}
            for k, total in vals.items():
                if k == "_count":
                    continue
                payload[f"perf/summary_total/{k}"] = total
                payload[f"perf/summary_avg/{k}"] = total / count
            self.log_event(f"{event_name}/summary", payload, metadata={"event_name": event_name})


def build_perf_logger(
    wandb_run: Any | None,
    dist_ctx: DistributedContext,
) -> PerfLogger:
    perf_logging_raw = os.getenv("PERF_LOGGING")
    enabled = wandb_run is not None if perf_logging_raw is None else perf_logging_raw == "1"
    include_non_main = os.getenv("PERF_LOG_NON_MAIN_RANKS", "0") == "1"
    gpu_sampling = os.getenv("PERF_GPU_SAMPLING", "0") == "1"
    return PerfLogger(
        wandb_run=wandb_run,
        dist_ctx=dist_ctx,
        enabled=enabled,
        include_non_main=include_non_main,
        gpu_sampling=gpu_sampling,
    )
