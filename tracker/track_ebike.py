"""电动自行车跟踪入口。从仓库根目录运行 python -m tracker.track_ebike。"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, closing
from dataclasses import asdict
from itertools import count
from pathlib import Path
from time import perf_counter

from .nsa import NSAConfig
from .records import read_cache, sha256_file
from .replay import ORUConfig


def write_json(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_tracks(frames, tracker, output, cache_output=None, profile=None) -> None:
    frames = iter(frames)
    for index in count():
        timing = profile.begin_frame(index) if profile is not None else None
        try:
            frame = next(frames)
        except StopIteration:
            break
        core_start = perf_counter() if timing is not None else 0.0
        result = tracker.update(frame)
        ready = perf_counter() if timing is not None else 0.0
        if cache_output is not None:
            if frame.frame_id == 0:
                write_json(cache_output, frame.source.cache_header())
            write_json(cache_output, frame.cache_record())
        if output is not None:
            write_json(output, result.output_record())
        if profile is not None:
            written = perf_counter() if timing is not None else 0.0
            profile.observe(frame.source.fps, result, timing, core_start, ready, written)
    if profile is not None:
        profile.finish_output((output, cache_output))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="逐帧回放检测缓存或对原始视频运行 NMS-Free 检测和跟踪")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--cache", type=Path, help="版本 1 的检测 JSONL")
    sources.add_argument("--video", type=Path, help="原始固定帧率视频")
    parser.add_argument("--weights", type=Path, help="视频模式使用的本地 NMS-Free 权重")
    parser.add_argument("--device", help="视频推理设备，例如 cpu 或 0")
    parser.add_argument("--output", type=Path, required=True, help="新建轨迹 JSONL")
    parser.add_argument("--save-cache", type=Path, help="视频模式额外导出检测 JSONL")
    parser.add_argument("--report", type=Path, help="新建性能报告 JSON")
    parser.add_argument("--warmup", type=int, default=0, help="报告排除的首段预热帧数")
    parser.add_argument("--profile-frames", type=int, default=600, help="分位数统计的最大采样帧数")
    parser.add_argument("--target-fps", type=float, help="视频基准的计划供帧速率，默认使用视频 FPS")
    parser.add_argument("--debug-predictions", action="store_true", help="附加带来源标记的丢失轨迹预测框")
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-age", type=int, default=15)
    parser.add_argument("--max-tracks", type=int, default=128)
    parser.add_argument("--max-detections", type=int, default=256)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--new-track-score", type=float, default=0.5)
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--ocm", choices=("off", "fixed", "adaptive"), default="off", help="方向关联模式")
    parser.add_argument("--ocm-weight", type=float, default=0.2, help="方向代价的最大权重")
    parser.add_argument("--ocm-interval", type=int, default=3, help="真实观测的目标间隔（帧）")
    parser.add_argument("--ocm-max-age", type=int, default=6, help="方向历史的最大帧龄")
    parser.add_argument("--ocm-min-displacement", type=float, default=2.0, help="历史方向的最小位移（像素）")
    parser.add_argument("--ocm-history-size", type=int, default=6, help="每轨最多保留的真实中心数")
    parser.add_argument("--ocr", action="store_true", help="对未匹配项执行最近实测框恢复")
    parser.add_argument("--ocr-max-age", type=int, default=6, help="恢复窗口（帧），受主链路 max-age 约束")
    parser.add_argument("--ocr-min-iou", type=float, default=0.3)
    parser.add_argument("--ocr-max-tracks", type=int, default=64)
    parser.add_argument("--ocr-max-detections", type=int, default=128)
    parser.add_argument("--oru", action="store_true", help="重关联后回放遮挡区间的滤波状态")
    parser.add_argument("--oru-max-missing", type=int, default=ORUConfig.max_missing, help="单轨最大缺帧数")
    parser.add_argument("--oru-max-steps", type=int, default=ORUConfig.max_steps, help="整帧最大虚拟更新尝试步数")
    parser.add_argument("--oru-noise-scale", type=float, default=ORUConfig.noise_scale, help="虚拟观测协方差倍率")
    parser.add_argument("--nsa", action="store_true", help="按真实检测置信度缩放观测协方差")
    parser.add_argument("--nsa-min-scale", type=float, default=NSAConfig.min_scale, help="NSA 的最小噪声倍率")
    args = parser.parse_args(argv)
    if args.video is not None and args.weights is None:
        parser.error("--video 需要 --weights")
    if args.cache is not None and any(x is not None for x in (args.weights, args.device, args.save_cache)):
        parser.error("--weights、--device 和 --save-cache 只用于视频模式")
    if args.target_fps is not None and (args.video is None or args.report is None):
        parser.error("--target-fps 需要视频模式和 --report")

    from .association import OCMConfig, OCRConfig
    from .core import Tracker, TrackingConfig

    try:
        config = TrackingConfig(
            min_hits=args.min_hits, max_age=args.max_age, max_tracks=args.max_tracks,
            max_detections=args.max_detections, min_score=args.min_score,
            new_track_score=args.new_track_score, min_iou=args.min_iou,
        )
        ocm = OCMConfig(
            mode=args.ocm, weight=args.ocm_weight, interval=args.ocm_interval, max_age=args.ocm_max_age,
            min_displacement=args.ocm_min_displacement, history_size=args.ocm_history_size,
        ) if args.ocm != "off" else None
        ocr = OCRConfig(
            max_age=args.ocr_max_age, min_iou=args.ocr_min_iou,
            max_tracks=args.ocr_max_tracks, max_detections=args.ocr_max_detections,
        ) if args.ocr else None
        oru = ORUConfig(args.oru_max_missing, args.oru_max_steps, args.oru_noise_scale) if args.oru else None
        nsa = NSAConfig(args.nsa_min_scale) if args.nsa else None
        tracker = Tracker(config, debug_predictions=args.debug_predictions, ocm=ocm, ocr=ocr, oru=oru, nsa=nsa)
        outputs = [path.resolve() for path in (args.output, args.save_cache, args.report) if path is not None]
        if len(set(outputs)) != len(outputs):
            raise ValueError("轨迹、检测缓存和报告需使用不同的输出路径")
        profile = None
        weights_sha256 = None
        if args.report is not None:
            from .timing import StreamProfile, environment

            profile = StreamProfile(args.warmup, args.profile_frames, paced=args.video is not None,
                                    target_fps=args.target_fps)
            if args.weights is not None:
                weights_sha256 = sha256_file(args.weights)
        with ExitStack() as stack:
            if args.cache is not None:
                stream = stack.enter_context(args.cache.open(encoding="utf-8-sig"))
                frames = read_cache(stream)
            else:
                from .video import load_detector, video_detections

                detect = load_detector(args.weights, config.min_score, args.device)
                frames = video_detections(args.video, detect)
            frames = stack.enter_context(closing(frames))
            output = stack.enter_context(args.output.open("x", encoding="utf-8"))
            cache_output = (
                stack.enter_context(args.save_cache.open("x", encoding="utf-8")) if args.save_cache else None
            )
            report_output = stack.enter_context(args.report.open("x", encoding="utf-8")) if args.report else None
            write_tracks(frames, tracker, output, cache_output, profile)
            if report_output is not None:
                input_path = args.video if args.video is not None else args.cache
                report = profile.report()
                source = tracker.source
                report.update(
                    schema_version=1, tracker_config=asdict(config), environment=environment(),
                    source=asdict(source), input_path=str(input_path), input_bytes=input_path.stat().st_size,
                    weights=str(args.weights) if args.weights is not None else None,
                    device_argument=args.device or ("auto" if args.video is not None else None),
                )
                if weights_sha256 is not None:
                    report["weights_sha256"] = weights_sha256
                if ocm is not None or ocr is not None:
                    report["association_config"] = {
                        "ocm": asdict(ocm) if ocm is not None else None,
                        "ocr": asdict(ocr) if ocr is not None else None,
                    }
                if oru is not None:
                    report["oru_config"] = asdict(oru)
                if nsa is not None:
                    report["nsa_config"] = asdict(nsa)
                write_json(report_output, report)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"跟踪失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
