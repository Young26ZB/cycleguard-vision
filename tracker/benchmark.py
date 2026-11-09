"""基于固定合成框重复测量跟踪核心，无模型或训练。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .association import OCMConfig, OCRConfig
from .core import Tracker, TrackingConfig
from .records import FrameDetections, ReplayStats, VideoSource
from .replay import ORUConfig
from .timing import StreamProfile, _latency_ms, environment
from .track_ebike import write_tracks


_MOTION_END = 5
_REAPPEARANCE = 10
_LAST_VISIBLE = 11


def _synthetic_frames(
    name: str, detections: int, spacing: int, count: int, fps: float, *, occlusion_period: int | None = None
):
    indexes = np.arange(detections)
    boxes = np.empty((detections, 6), dtype=np.float64)
    boxes[:, 0] = (indexes % 32) * spacing + 10
    boxes[:, 1] = (indexes // 32) * (180 if occlusion_period else spacing) + 10
    boxes[:, 2] = boxes[:, 0] + 42
    boxes[:, 3] = boxes[:, 1] + 56
    boxes[:, 4] = 0.9
    boxes[:, 5] = 0
    height_margin = 112 if occlusion_period else 32
    source = VideoSource(name, fps, int(boxes[:, 2].max()) + 32, int(boxes[:, 3].max()) + height_margin)
    for index in range(count):
        phase = index % occlusion_period if occlusion_period else 0
        if occlusion_period and (_MOTION_END < phase < _REAPPEARANCE or phase > _LAST_VISIBLE):
            rows = np.empty((0, 6))
        else:
            rows = boxes.copy()
            if occlusion_period:
                rows[:, (1, 3)] += min(phase, _MOTION_END) * 16
            else:
                rows[:, (0, 2)] += index % 6
        yield FrameDetections(source, index, index / fps, rows)


def benchmark(config: TrackingConfig, frames: int, warmup: int, repeat: int, fps: float) -> dict:
    ordinary_count = min(12, config.max_tracks, config.max_detections)
    dense_count = min(config.max_tracks, config.max_detections)
    # 压力帧始终超过检测容量，并记录轨迹池的拒绝数。
    cases = (
        ("ordinary", ordinary_count, 64),
        ("dense", dense_count, 12),
        ("over_capacity", config.max_detections + 16, 12),
    )
    reports = {}
    for name, count, spacing in cases:
        runs = []
        for _ in range(repeat):
            profile = StreamProfile(warmup, frames, target_fps=fps)
            inputs = _synthetic_frames(name, count, spacing, warmup + frames, fps)
            tracker = Tracker(config)
            write_tracks(inputs, tracker, None, profile=profile)
            runs.append(profile.report())
        reports[name] = {
            "detections_per_frame": count, "spacing_px": spacing,
            "frame_size_px": [tracker.source.width, tracker.source.height], "runs": runs,
        }
    return {
        "schema_version": 1,
        "benchmark": "ebike_tracking_synthetic",
        "scope": "tracking_core_only",
        "device": "cpu",
        "environment": environment(),
        "tracker_config": asdict(config),
        "profile": {"frames": frames, "warmup": warmup, "repeat": repeat, "target_fps": fps},
        "cases": reports,
    }


class _ReappearanceProfile(StreamProfile):
    def __init__(self, warmup, sample_limit, fps, occlusion_period):
        super().__init__(warmup, sample_limit, target_fps=fps)
        self.occlusion_period = occlusion_period
        self.reappearance_samples = []
        self.reappearance_replay_steps = 0
        self.reappearance_replay_fallbacks = 0

    def observe(self, fps, result, timing, core_start, ready, written):
        super().observe(fps, result, timing, core_start, ready, written)
        if (
            self.occlusion_period and timing is not None
            and len(self.reappearance_samples) < self.sample_limit
            and result.frame_id % self.occlusion_period == _REAPPEARANCE
        ):
            self.reappearance_samples.append(ready - core_start)
            if result.replay is not None:
                self.reappearance_replay_steps += result.replay.oru_steps
                self.reappearance_replay_fallbacks += result.replay.oru_fallbacks

    def report(self):
        report = super().report()
        report["reappearance_latency_ms"] = _latency_ms(self.reappearance_samples)
        report["reappearance_samples"] = len(self.reappearance_samples)
        return report


def association_benchmark(config: TrackingConfig, frames: int, warmup: int, repeat: int, fps: float) -> dict:
    fixed, adaptive, ocr = OCMConfig(), OCMConfig(mode="adaptive"), OCRConfig()
    profiles = (
        ("off", None, None),
        ("ocm_fixed", fixed, None),
        ("ocr", None, ocr),
        ("ocm_fixed_ocr", fixed, ocr),
        ("ocm_adaptive", adaptive, None),
        ("ocm_adaptive_ocr", adaptive, ocr),
    )
    dense = min(config.max_tracks, config.max_detections)
    scenarios = (
        ("ordinary", min(12, dense), 64, False),
        ("dense", dense, 12, False),
        ("dense_misses", dense, 12, True),
    )
    # 隔开重现事件，使上一轮的轨迹自然到期，每轮都经历一次预测漂移。
    occlusion_period = _LAST_VISIBLE + config.max_age + 1
    cases = {}
    for name, detections, spacing, misses in scenarios:
        period = occlusion_period if misses else None
        runs = {mode: [] for mode, _, _ in profiles}
        for run in range(repeat):
            # 轮换起始配置，减少运行顺序对尾延迟的影响。
            offset = run % len(profiles)
            for mode, momentum, recovery in profiles[offset:] + profiles[:offset]:
                profile = _ReappearanceProfile(warmup, frames, fps, period)
                tracker = Tracker(config, ocm=momentum, ocr=recovery)
                inputs = _synthetic_frames(name, detections, spacing, warmup + frames, fps, occlusion_period=period)
                write_tracks(inputs, tracker, None, profile=profile)
                runs[mode].append(profile.report())
        cases[name] = {
            "detections_per_visible_frame": detections,
            "spacing_px": spacing,
            "frame_size_px": [tracker.source.width, tracker.source.height],
            "runs": runs,
        }
    return {
        "schema_version": 1,
        "benchmark": "observation_centric_association",
        "scope": "tracking_core_only",
        "device": "cpu",
        "environment": environment(),
        "tracker_config": asdict(config),
        "profiles": {
            name: {"ocm": asdict(momentum) if momentum else None, "ocr": asdict(recovery) if recovery else None}
            for name, momentum, recovery in profiles
        },
        "sampling": {
            "frames": frames, "warmup": warmup, "repeat": repeat, "target_fps": fps,
            "occlusion_period": occlusion_period, "reappearance_phase": _REAPPEARANCE,
        },
        "cases": cases,
    }


def oru_benchmark(
    config: TrackingConfig, frames: int, warmup: int, repeat: int, fps: float, replay: ORUConfig
) -> dict:
    """用同一批同步重现帧对比 ORU 的开销和预算回退。"""
    momentum, recovery = OCMConfig(mode="adaptive"), OCRConfig()
    profiles = (("off", None), ("on", replay))
    detections = min(config.max_tracks, config.max_detections)
    period = _LAST_VISIBLE + config.max_age + 1
    runs = {"off": [], "on": []}
    replay_keys = vars(ReplayStats())
    for run in range(repeat):
        ordered = profiles if run % 2 == 0 else profiles[::-1]
        for mode, replay_config in ordered:
            tracker = Tracker(config, ocm=momentum, ocr=recovery, oru=replay_config)
            inputs = _synthetic_frames("oru_dense_misses", detections, 12, warmup + frames, fps, occlusion_period=period)
            profile = _ReappearanceProfile(warmup, frames, fps, period)
            write_tracks(inputs, tracker, None, profile=profile)
            report = profile.report()
            for field in ("workload_totals", "workload_peaks"):
                for key in replay_keys:
                    report[field].setdefault(key, 0)
            report["reappearance_replay_steps"] = profile.reappearance_replay_steps
            report["reappearance_replay_fallbacks"] = profile.reappearance_replay_fallbacks
            runs[mode].append(report)
    return {
        "schema_version": 1,
        "benchmark": "occlusion_filter_replay",
        "scope": "tracking_core_only",
        "device": "cpu",
        "environment": environment(),
        "tracker_config": asdict(config),
        "association_config": {"ocm": asdict(momentum), "ocr": asdict(recovery)},
        "oru_config": asdict(replay),
        "sampling": {
            "frames": frames, "warmup": warmup, "repeat": repeat, "target_fps": fps,
            "occlusion_period": period, "reappearance_phase": _REAPPEARANCE,
            "detections_per_visible_frame": detections,
            "frame_size_px": [tracker.source.width, tracker.source.height],
        },
        "runs": runs,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成跟踪核心的可复现合成基准")
    parser.add_argument("--output", type=Path, required=True, help="新建基准 JSON")
    parser.add_argument("--frames", type=int, default=240, help="每组每次的采样帧数")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--target-fps", type=float, default=30)
    parser.add_argument("--max-tracks", type=int, default=TrackingConfig.max_tracks)
    parser.add_argument("--max-detections", type=int, default=TrackingConfig.max_detections)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--association", action="store_true", help="对比 OCM/OCR 开关与密集遮挡重现")
    modes.add_argument("--oru", action="store_true", help="对比同步重现场景的 ORU 开关")
    parser.add_argument("--oru-max-missing", type=int, default=ORUConfig.max_missing)
    parser.add_argument("--oru-max-steps", type=int, default=ORUConfig.max_steps)
    parser.add_argument("--oru-noise-scale", type=float, default=ORUConfig.noise_scale)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat 必须为正")
    try:
        config = TrackingConfig(max_tracks=args.max_tracks, max_detections=args.max_detections)
        sampling = (config, args.frames, args.warmup, args.repeat, args.target_fps)
        if args.oru:
            report = oru_benchmark(*sampling, ORUConfig(args.oru_max_missing, args.oru_max_steps, args.oru_noise_scale))
        else:
            measure = association_benchmark if args.association else benchmark
            report = measure(*sampling)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"基准失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
