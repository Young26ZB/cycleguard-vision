"""统一记录跟踪核心与原始视频链路的计时和工作量。"""

from __future__ import annotations

import os
import platform
import sys
from itertools import chain
from math import floor, isfinite
from time import perf_counter, sleep

import numpy as np
import psutil
import scipy

from .records import FrameTracks


def environment() -> dict:
    torch = sys.modules.get("torch")
    cv2 = sys.modules.get("cv2")
    info = {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__ if torch is not None else None,
        "opencv": cv2.__version__ if cv2 is not None else None,
        "torch_threads": torch.get_num_threads() if torch is not None else None,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    if torch is not None and torch.cuda.is_initialized():
        info["cuda_runtime"] = torch.version.cuda
        info["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return info


def _latency_ms(samples: list[float]) -> dict | None:
    if not samples:
        return None
    values = np.percentile(samples, [50, 95, 99]) * 1000
    return dict(zip(("p50", "p95", "p99"), values.tolist()))


class StreamProfile:
    """用有界样本记录分位数，并对整个测量区间汇总工作量。"""

    def __init__(
        self, warmup: int, sample_limit: int, *, paced: bool = False, target_fps: float | None = None
    ) -> None:
        if warmup < 0 or sample_limit < 1:
            raise ValueError("预热帧数须非负，采样上限须为正")
        if target_fps is not None and (not isfinite(target_fps) or target_fps <= 0):
            raise ValueError("目标帧率必须为有限正数")
        self.warmup = warmup
        self.sample_limit = sample_limit
        self.paced = paced
        self.target_fps = target_fps
        self._origin = None
        self._last_ready = None
        self._seen = 0
        self._count = 0
        self._core_seconds = 0.0
        self._sink_seconds = 0.0
        self._flush_seconds = 0.0
        self._core = []
        self._end_to_end = []
        self._sink = []
        self._totals = {}
        self._peaks = {}
        self._overruns = 0
        self._late_frames = 0
        self._last_backlog = 0
        self._peak_backlog = 0
        self._last_lag = 0.0
        self._process = psutil.Process()
        self._peak_rss = self._process.memory_info().rss

    def begin_frame(self, index: int) -> tuple[float, float] | None:
        if index < self.warmup:
            return None
        started = perf_counter()
        if self._origin is None:
            self._origin = started
        if self.paced and self.target_fps is not None:
            arrival = self._origin + (index - self.warmup) / self.target_fps
            if started < arrival:
                sleep(arrival - started)
                started = perf_counter()
        else:
            arrival = started
        return arrival, started

    def observe(
        self, fps: float, result: FrameTracks, timing: tuple[float, float] | None,
        core_start: float, ready: float, written: float
    ) -> None:
        if self.target_fps is None:
            self.target_fps = fps
        self._seen += 1
        if timing is None:
            return
        arrival, started = timing
        core = ready - core_start
        sink = written - ready
        self._count += 1
        self._core_seconds += core
        self._sink_seconds += sink
        self._last_ready = ready
        frame_period = 1.0 / self.target_fps
        self._overruns += core > frame_period
        if self.paced:
            self._late_frames += ready - arrival > frame_period
            self._last_lag = max(0.0, started - arrival)
            self._last_backlog = floor(self._last_lag * self.target_fps)
            self._peak_backlog = max(self._peak_backlog, self._last_backlog)

        if len(self._core) < self.sample_limit:
            self._core.append(core)
            self._sink.append(sink)
            if self.paced:
                self._end_to_end.append(ready - arrival)
        association = vars(result.association).items() if result.association is not None else ()
        replay = vars(result.replay).items() if result.replay is not None else ()
        nsa = vars(result.nsa).items() if result.nsa is not None else ()
        for key, value in chain(vars(result.stats).items(), association, replay, nsa, (("real_updates", result.real_updates),)):
            self._totals[key] = self._totals.get(key, 0) + value
            self._peaks[key] = max(self._peaks.get(key, 0), value)
        if self._count % 32 == 1:
            self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)

    def finish_output(self, streams: tuple) -> None:
        start = perf_counter()
        for stream in streams:
            if stream is not None:
                stream.flush()
        self._flush_seconds = perf_counter() - start

    def report(self) -> dict:
        memory = self._process.memory_info()
        observed_peak = max(self._peak_rss, memory.rss)
        if hasattr(memory, "peak_wset"):
            process_peak = memory.peak_wset
        else:
            import resource

            process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != "darwin":
                process_peak *= 1024
        elapsed = self._last_ready - self._origin if self._count else 0.0
        period_ms = 1000 / self.target_fps if self.target_fps else None
        return {
            "mode": "paced_video" if self.paced else "tracking_core_only",
            "frames_processed": self._seen,
            "warmup_frames": self.warmup,
            "measured_frames": self._count,
            "latency_samples": len(self._core),
            "sample_limit": self.sample_limit,
            "target_fps": self.target_fps,
            "frame_period_ms": period_ms,
            "suggested_tracking_budget_ms": 0.20 * period_ms if period_ms else None,
            "tracking_latency_ms": _latency_ms(self._core),
            "end_to_end_latency_ms": _latency_ms(self._end_to_end),
            "tracking_throughput_fps": self._count / self._core_seconds if self._core_seconds else None,
            "observed_throughput_fps": self._count / elapsed if elapsed else None,
            "tracking_over_frame_period": self._overruns,
            "missed_frame_deadlines": self._late_frames if self.paced else None,
            "peak_schedule_backlog_frames": self._peak_backlog if self.paced else None,
            "final_schedule_backlog_frames": self._last_backlog if self.paced else None,
            "final_start_lag_ms": self._last_lag * 1000 if self.paced else None,
            "dropped_frames": 0 if self.paced else None,
            "output": {
                "per_frame_latency_ms": _latency_ms(self._sink),
                "total_ms": self._sink_seconds * 1000,
                "final_flush_ms": self._flush_seconds * 1000,
                "rendering_and_encoding": "disabled",
            },
            "peak_sampled_rss_bytes": observed_peak,
            "process_peak_rss_bytes": process_peak,
            "workload_totals": self._totals,
            "workload_peaks": self._peaks,
        }
