"""复核清单与电动自行车诊断计分。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path, PurePosixPath
from statistics import median

from PIL import Image


REVIEW_REASONS = ("uncertain_class", "uncertain_bounds", "uncertain_ownership")
LIGHTING = ("day", "night", "unknown")
OCCLUSION = ("none", "partial", "unknown")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path):
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    return width, height


def rooted_image(root, name):
    """验证跨平台相对路径，并限制源文件位于数据根下。"""
    if not isinstance(name, str) or not name or any(c in name for c in "\\:\0"):
        raise ValueError(f"无效的相对图片路径 {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts or not path.suffix:
        raise ValueError(f"无效的相对图片路径 {name!r}")
    root = Path(root).resolve()
    source = (root / path).resolve()
    if root not in source.parents or not source.is_file():
        raise ValueError(f"图片不存在或超出数据根目录 {name!r}")
    return source


def validate_box(box, width, height, context):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in box
    ):
        raise ValueError(f"{context} 的框必须包含 4 个有限数值")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{context} 的框超出图像或面积为零 {box}")
    return tuple(box)


def _object(value, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} 必须是对象")
    return value


def _list(record, key, context):
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context} 必须显式提供 {key} 列表")
    return value


def _unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"诊断清单中有重复字段 {key!r}")
        result[key] = value
    return result


def load_manifest(path, data_root):
    manifest = _object(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys), "诊断清单")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("诊断清单 schema_version 必须为 1")
    if not isinstance(manifest.get("dataset_version"), str) or not manifest["dataset_version"].strip():
        raise ValueError("诊断清单缺少 dataset_version")
    images = _object(manifest.get("images"), "images")
    if not images:
        raise ValueError("诊断清单缺少图片")

    seen = set()
    for name, record in images.items():
        source = rooted_image(data_root, name)
        if source in seen:
            raise ValueError(f"重复的诊断图片路径 {name!r}")
        seen.add(source)
        _object(record, name)
        width, height = image_size(source)
        if any(type(record.get(key)) is not int for key in ("width", "height")) or (
            record["width"], record["height"]
        ) != (width, height):
            raise ValueError(f"{name} 的尺寸与原图不符")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest) or digest != sha256_file(source):
            raise ValueError(f"{name} 的 SHA-256 与原图不符")
        if type(record.get("training_overlap")) is not bool:
            raise ValueError(f"{name} 必须声明 training_overlap")
        if record.get("lighting") not in LIGHTING:
            raise ValueError(f"{name} 的 lighting 无效")
        status = record.get("status")
        reasons = _list(record, "review", name)
        if status not in ("ready", "pending"):
            raise ValueError(f"{name} 的复核状态无效")
        for index, item in enumerate(reasons, 1):
            context = f"{name} 的复核项 {index}"
            _object(item, context)
            if item.get("reason") not in REVIEW_REASONS:
                raise ValueError(f"{context} 原因无效")
            for field in ("location", "evidence"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"{context} 缺少 {field}")
            if "conclusion" not in item or (
                item["conclusion"] is not None
                and (not isinstance(item["conclusion"], str) or not item["conclusion"].strip())
            ):
                raise ValueError(f"{context} 结论无效")
        if (status == "pending") != any(item["conclusion"] is None for item in reasons):
            raise ValueError(f"{name} 的复核状态与未决原因不一致")
        for index, target in enumerate(_list(record, "targets", name), 1):
            context = f"{name} 的真实框 {index}"
            _object(target, context)
            validate_box(target.get("box"), width, height, context)
            if target.get("occlusion") not in OCCLUSION:
                raise ValueError(f"{context} 缺少有效的遮挡状态")
        for index, box in enumerate(_list(record, "ignore", name), 1):
            validate_box(box, width, height, f"{name} 的忽略框 {index}")
    return manifest


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    intersection = _intersection(a, b)
    return intersection / (_area(a) + _area(b) - intersection)


def _best_overlap(overlaps, indices, threshold):
    index = max(indices, key=lambda i: (overlaps[i], -i), default=None)
    if index is not None and overlaps[index] >= threshold:
        return index
    return None


def validate_thresholds(confidence, match_iou, ignore_ioa):
    for name, value in (("confidence", confidence), ("match_iou", match_iou), ("ignore_ioa", ignore_ioa)):
        if type(value) not in (float, int) or not 0 < value <= 1:
            raise ValueError(f"{name} 必须在 (0, 1] 内")


def _score_frame(name, record, predictions, confidence, match_iou, ignore_ioa):
    targets = [target["box"] for target in record["targets"]]
    candidates = []
    for index, prediction in enumerate(predictions):
        context = f"{name} 的预测 {index + 1}"
        box = validate_box(prediction.get("box"), record["width"], record["height"], context)
        score = prediction.get("confidence")
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError(f"{context} 的置信度无效")
        if score >= confidence:
            candidates.append(dict(index=index, box=box, confidence=score))
    candidates.sort(key=lambda prediction: (-prediction["confidence"], prediction["index"]))

    unmatched = set(range(len(targets)))
    remaining = []
    matched_ious = []
    for candidate in candidates:
        overlaps = [_iou(candidate["box"], target) for target in targets]
        target = _best_overlap(overlaps, unmatched, match_iou)
        if target is None:
            remaining.append((candidate, overlaps))
        else:
            unmatched.remove(target)
            candidate.update(outcome="tp", target=target, iou=overlaps[target])
            matched_ious.append(overlaps[target])

    matched = set(range(len(targets))) - unmatched
    for candidate, overlaps in remaining:
        duplicate = _best_overlap(overlaps, matched, match_iou)
        if duplicate is not None:
            candidate.update(outcome="duplicate", target=duplicate, iou=overlaps[duplicate])
            continue
        box = candidate["box"]
        ioas = [_intersection(box, region) / _area(box) for region in record["ignore"]]
        region = _best_overlap(ioas, range(len(ioas)), ignore_ioa)
        if region is not None:
            candidate.update(outcome="ignored", region=region, ioa=ioas[region])
        else:
            candidate.update(outcome="fp")

    counts = Counter(candidate["outcome"] for candidate in candidates)
    frame = {
        "status": "ready", "lighting": record["lighting"], "target_count": len(targets),
        "max_target_iou": max((_iou(a, b) for a, b in combinations(targets, 2)), default=None),
        "targets": [
            {**target, "relative_area": _area(target["box"]) / (record["width"] * record["height"])}
            for target in record["targets"]
        ],
        "ignore": record["ignore"], "predictions": candidates, "unmatched_targets": sorted(unmatched),
        "tp": counts["tp"], "fp": counts["fp"] + counts["duplicate"], "fn": len(unmatched),
        "duplicates": counts["duplicate"], "ignored": counts["ignored"],
    }
    return frame, matched_ious


def score_diagnostics(manifest, predictions, *, confidence=0.25, match_iou=0.50, ignore_ioa=0.80):
    """按已验证清单计分，预测按原图路径分组且已筛选为 ebike 类。"""
    validate_thresholds(confidence, match_iou, ignore_ioa)
    images = manifest["images"]
    ready = {name for name, record in images.items() if record["status"] == "ready"}
    missing, extra = ready - predictions.keys(), predictions.keys() - images.keys()
    if missing or extra:
        raise ValueError(f"预测与诊断清单不对应，缺 {sorted(missing)}，多 {sorted(extra)}")
    counts, frames, ious = Counter(), {}, []
    for name, record in images.items():
        if record["status"] == "pending":
            frames[name] = {"status": "pending", "lighting": record["lighting"], "review": record["review"]}
            continue
        frame, matched_ious = _score_frame(name, record, predictions[name], confidence, match_iou, ignore_ioa)
        frames[name] = frame
        ious.extend(matched_ious)
        counts.update({key: frame[key] for key in ("tp", "fp", "fn", "duplicates", "ignored", "target_count")})

    scored_predictions = counts["tp"] + counts["fp"]
    overlap_images = [name for name, record in images.items() if record["training_overlap"]]
    return {
        "purpose": "diagnostic_only",
        "dataset_version": manifest["dataset_version"],
        "training_overlap": bool(overlap_images),
        "training_overlap_images": overlap_images,
        "thresholds": {"confidence": confidence, "match_iou": match_iou, "ignore_ioa": ignore_ioa},
        "image_count": len(images), "scored_images": len(ready), "pending_images": len(images) - len(ready),
        **{key: counts[key] for key in ("target_count", "tp", "fp", "fn", "duplicates", "ignored")},
        "precision": counts["tp"] / scored_predictions if scored_predictions else None,
        "recall": counts["tp"] / counts["target_count"] if counts["target_count"] else None,
        "duplicate_rate": counts["duplicates"] / scored_predictions if scored_predictions else None,
        "localization_error_median": median(1 - iou for iou in ious) if ious else None,
        "localization_pair_count": len(ious),
        "frames": frames,
    }
