"""离线运行冻结的跟踪消融协议。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from itertools import zip_longest
from math import isfinite
from pathlib import Path

import numpy as np

from .association import OCMConfig, OCRConfig
from .core import Tracker, TrackingConfig
from .evaluation import GroundTruth, MOTSequence, RecoveryMetrics
from .nsa import NSAConfig
from .profiles import ablation_profiles
from .records import read_cache, sha256_file
from .replay import ORUConfig


@dataclass(frozen=True)
class Sequence:
    name: str
    split: str
    cache: Path
    truth: Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_manifest(path: Path) -> tuple[str, list[Sequence]]:
    manifest = _read_json(path)
    weights = manifest["weights_sha256"]
    if not isinstance(weights, str) or not re.fullmatch(r"[0-9a-f]{64}", weights):
        raise ValueError("weights_sha256 必须是权重文件的小写 SHA-256")
    sequences = [
        Sequence(row["id"], row["split"], (path.parent / row["cache"]).resolve(), (path.parent / row["truth"]).resolve())
        for row in manifest["sequences"]
    ]
    if {seq.split for seq in sequences} != {"validation", "test"}:
        raise ValueError("清单须包含独立的 validation 和 test 视频")
    if any(not isinstance(seq.name, str) or not seq.name for seq in sequences):
        raise ValueError("序列 ID 必须是非空字符串")
    if len({seq.name for seq in sequences}) != len(sequences):
        raise ValueError("每个序列 ID 只能出现在一个划分中")
    if len({p for seq in sequences for p in (seq.cache, seq.truth)}) != 2 * len(sequences):
        raise ValueError("各序列不能共享检测缓存或真值文件")
    return weights, sequences


def _check_protocol(protocol: dict) -> None:
    iou = protocol["iou"]
    bins = protocol["occlusion_bins"]
    if type(iou) not in (int, float) or not 0 < iou <= 1:
        raise ValueError("iou 须在 (0, 1] 内")
    if not isinstance(bins, list) or any(type(x) is not int or x < 1 for x in bins) or bins != sorted(set(bins)):
        raise ValueError("occlusion_bins 须是严格递增的正整数列表")
    for key in ("target_fps", "tracking_p95_ms"):
        value = protocol[key]
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError(f"{key} 须为有限正数")
    gates = protocol["quality_thresholds"]
    for key in ("min_idf1", "min_assa", "min_same_id_rate", "max_wrong_reconnect_rate"):
        if type(gates[key]) not in (int, float) or not 0 <= gates[key] <= 1:
            raise ValueError(f"{key} 须在 [0, 1] 内")
    if type(gates["max_id_switches"]) is not int or gates["max_id_switches"] < 0:
        raise ValueError("max_id_switches 须为非负整数")


def _mot_metrics(iou: float) -> tuple:
    try:
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError as exc:
        raise ValueError("标准 MOT 指标需要兼容当前 NumPy 的 TrackEval 环境。仅验证事件逻辑可用 --events-only") from exc
    config = {"THRESHOLD": iou, "PRINT_CONFIG": False}
    return Identity(config), HOTA(), CLEAR(config)


def _evaluate_profile(sequences, config, options, protocol, metrics):
    recovery = RecoveryMetrics(protocol["iou"], tuple(protocol["occlusion_bins"]), config.min_score)
    mot_results = [{} for _ in metrics]
    input_info = {}
    for sequence in sequences:
        tracker = Tracker(config, **options)
        recovery.reset_sequence()
        mot = MOTSequence() if metrics else None
        count = 0
        with sequence.cache.open(encoding="utf-8-sig") as cache, sequence.truth.open(encoding="utf-8-sig") as truth:
            for frame, line in zip_longest(read_cache(cache), truth):
                if frame is None or line is None:
                    raise ValueError(f"{sequence.name} 的真值和缓存帧数不一致")
                annotations = GroundTruth.from_record(json.loads(line), frame)
                result = tracker.update(frame)
                ious = recovery.observe(annotations, frame, result)
                if mot is not None:
                    mot.append(annotations, result, ious)
                count += 1
        input_info[sequence.name] = {"frames": count, "source": asdict(tracker.source)}
        if mot is not None:
            data = mot.data()
            for metric, results in zip(metrics, mot_results):
                results[sequence.name] = metric.eval_sequence(data)
    standard = None
    if metrics:
        identity, hota, clear = (metric.combine_sequences(results) for metric, results in zip(metrics, mot_results))
        standard = {
            "idf1": float(identity["IDF1"]), "assa": float(np.mean(hota["AssA"])), "id_switches": int(clear["IDSW"])
        }
    return {"mot": standard, "recovery": recovery.report()}, input_info, recovery.calibration_report()


def evaluate(
    manifest_path: Path, protocol_path: Path, split: str, *, events_only: bool = False,
    validation_path: Path | None = None, timing_path: Path | None = None,
) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("split 必须为 validation 或 test")
    if timing_path is not None and (split != "test" or events_only):
        raise ValueError("配置选择需要独立测试集的标准指标和目标设备计时")
    weights, sequences = _load_manifest(manifest_path)
    protocol = _read_json(protocol_path)
    _check_protocol(protocol)
    config = TrackingConfig(**protocol.get("tracking", {}))
    profiles = ablation_profiles(
        NSAConfig(protocol.get("nsa_min_scale", NSAConfig.min_scale)),
        ocm=OCMConfig(**protocol.get("ocm", {})),
        ocr=OCRConfig(**protocol.get("ocr", {})),
        oru=ORUConfig(**protocol.get("oru", {})),
    )
    registration = {
        "weights_sha256": weights,
        "protocol_sha256": sha256_file(protocol_path),
        "sequences": {
            seq.name: {"split": seq.split, "cache_sha256": sha256_file(seq.cache), "truth_sha256": sha256_file(seq.truth)}
            for seq in sequences
        },
        "tracking": asdict(config),
        "profiles": {name: {key: asdict(value) for key, value in options.items()} for name, options in profiles.items()},
    }
    cache_hashes = [row["cache_sha256"] for row in registration["sequences"].values()]
    if len(set(cache_hashes)) != len(cache_hashes):
        raise ValueError("各序列须使用不同内容的检测缓存")
    if split == "test":
        if validation_path is None:
            raise ValueError("测试阶段需要验证集报告以核对已冻结协议")
        validation = _read_json(validation_path)
        if (validation["split"], validation["events_only"], validation["registration"]) != (
            "validation", events_only, registration
        ):
            raise ValueError("验证报告中的数据、权重、指标模式或配置与测试阶段不一致")
    metrics = () if events_only else _mot_metrics(protocol["iou"])
    subset = [seq for seq in sequences if seq.split == split]
    results = {}
    for name, options in profiles.items():
        result, input_info, calibration = _evaluate_profile(subset, config, options, protocol, metrics)
        results[name] = result
    report = {
        "schema_version": 1, "scope": "annotated_detection_cache", "split": split, "events_only": events_only,
        "registration": registration, "inputs": input_info, "detector_calibration": calibration, "profiles": results,
    }
    if timing_path is not None:
        report["deployment"] = select_realtime_profile(report, protocol, timing_path)
    return report


def select_realtime_profile(report: dict, protocol: dict, timing_path: Path) -> dict:
    timings = _read_json(timing_path)
    setup = {key: timings[key] for key in ("environment", "device_argument", "warmup_frames", "sample_limit")}
    if not setup["environment"] or not setup["device_argument"]:
        raise ValueError("计时清单须声明目标环境和推理设备")
    gates = protocol["quality_thresholds"]
    target = protocol["target_fps"]
    evidence = {}
    qualified = {}
    for name, result in report["profiles"].items():
        failures = []
        mot = result["mot"]
        recovery = result["recovery"]["all"]["all_events"]
        checks = {
            "idf1": mot is not None and mot["idf1"] >= gates["min_idf1"],
            "assa": mot is not None and mot["assa"] >= gates["min_assa"],
            "id_switches": mot is not None and mot["id_switches"] <= gates["max_id_switches"],
            "same_id_rate": recovery["same_id_rate"] is not None and recovery["same_id_rate"] >= gates["min_same_id_rate"],
            "wrong_reconnect_rate": recovery["wrong_reconnect_rate"] is not None
            and recovery["wrong_reconnect_rate"] <= gates["max_wrong_reconnect_rate"],
        }
        failures.extend(f"quality:{key}" for key, passed in checks.items() if not passed)
        paths = timings["profiles"].get(name, {})
        latency = {}
        timing_reports = {}
        for sequence, info in report["inputs"].items():
            if sequence not in paths:
                failures.append(f"{sequence}:missing_paced_video")
                continue
            path = (timing_path.parent / paths[sequence]).resolve()
            timing_reports[sequence] = str(path)
            try:
                run = _read_json(path)
            except (OSError, ValueError):
                failures.append(f"{sequence}:unreadable_paced_video")
                continue
            if not isinstance(run, dict):
                failures.append(f"{sequence}:invalid_paced_video")
                continue
            if any(run.get(key) != value for key, value in setup.items()):
                failures.append(f"{sequence}:incompatible_measurement_setup")
                continue
            issue = _timing_failure(run, report["registration"], name, info, target, protocol["tracking_p95_ms"])
            if issue:
                failures.append(f"{sequence}:{issue}")
                continue
            latency[sequence] = run["end_to_end_latency_ms"]["p95"]
        evidence[name] = {"failures": failures, "end_to_end_p95_ms_by_sequence": latency, "timing_reports": timing_reports}
        if not failures:
            qualified[name] = max(latency.values())
    selected = min(qualified, key=lambda name: (qualified[name], name)) if qualified else None
    return {"selected": selected, "ranking": "minimize_worst_sequence_end_to_end_p95", "measurement_setup": setup,
            "profiles": evidence}


def _timing_failure(run, registration, profile, sequence, target_fps, tracking_budget):
    options = registration["profiles"][profile]
    association = run.get("association_config") or {}
    actual = {
        key: value for key, value in {
            "ocm": association.get("ocm"), "ocr": association.get("ocr"),
            "oru": run.get("oru_config"), "nsa": run.get("nsa_config"),
        }.items() if value is not None
    }
    if (
        run.get("mode") != "paced_video" or run.get("target_fps") != target_fps
        or run.get("weights_sha256") != registration["weights_sha256"]
        or run.get("tracker_config") != registration["tracking"] or actual != options
        or run.get("source") != sequence["source"] or run.get("frames_processed") != sequence["frames"]
    ):
        return "incompatible_paced_video"
    if run.get("dropped_frames") != 0 or run.get("final_schedule_backlog_frames") != 0:
        return "drops_or_backlog"
    throughput = run.get("observed_throughput_fps")
    if type(throughput) not in (int, float) or not isfinite(throughput) or throughput < target_fps:
        return "throughput"
    core, total = run.get("tracking_latency_ms"), run.get("end_to_end_latency_ms")
    if not all(_valid_percentiles(item) for item in (core, total)):
        return "missing_latency"
    if core["p95"] > tracking_budget or total["p95"] > 1000 / target_fps:
        return "latency_budget"
    return None


def _valid_percentiles(item) -> bool:
    if not isinstance(item, dict):
        return False
    values = [item.get(key) for key in ("p50", "p95", "p99")]
    return all(type(value) in (int, float) and isfinite(value) and value >= 0 for value in values) and values == sorted(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="用同一检测缓存运行跟踪质量消融")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--timing-manifest", type=Path)
    parser.add_argument("--events-only", action="store_true", help="只评估遮挡事件及检测分数分桶")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            args.manifest, args.protocol, args.split, events_only=args.events_only,
            validation_path=args.validation_report, timing_path=args.timing_manifest,
        )
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"评估失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
