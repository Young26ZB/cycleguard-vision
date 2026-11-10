"""按真实遮挡事件评价跟踪恢复，并复用标准 MOT 指标。"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ultralytics.utils.metrics import bbox_ioa

from .core import _assign_iou
from .records import FrameDetections, FrameTracks


@dataclass(frozen=True)
class GroundTruth:
    ids: tuple[int, ...]
    visible: np.ndarray
    occluded: frozenset[int]
    partial: frozenset[int]

    @classmethod
    def from_record(cls, record: dict, frame: FrameDetections) -> GroundTruth:
        if type(record["frame_id"]) is not int or record["frame_id"] != frame.frame_id:
            raise ValueError(f"真值帧号与检测缓存不一致：{frame.frame_id}")
        rows = record["visible"]
        if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != 6 for row in rows):
            raise ValueError("visible 必须是 [ID, x1, y1, x2, y2, class_id] 列表")
        ids = tuple(row[0] for row in rows)
        visible_ids = _checked_ids(ids)
        occluded = _checked_ids(record["occluded"])
        partial = _checked_ids(record.get("partial", []))
        if visible_ids & occluded or not partial <= visible_ids:
            raise ValueError("同一帧的全遮挡 ID 与可见 ID 须互斥，部分遮挡 ID 须可见")
        boxes = FrameDetections(
            frame.source, frame.frame_id, frame.timestamp_s, [[*row[1:5], 1, row[5]] for row in rows]
        )
        return cls(ids, boxes.detections, occluded, partial)


def _checked_ids(values) -> frozenset[int]:
    if not isinstance(values, (list, tuple)) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("真值 ID 须为非负整数列表")
    unique = frozenset(values)
    if len(unique) != len(values):
        raise ValueError("同一帧中的真值 ID 须唯一")
    return unique


def _iou(truth: np.ndarray, other: np.ndarray) -> np.ndarray:
    scores = bbox_ioa(truth[:, :4], other[:, :4], iou=True, eps=0.0)
    scores[truth[:, 5, None] != other[None, :, 5]] = 0
    return scores


def _occlusion_labels(bounds: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{start}-{end}" for start, end in zip((1, *(end + 1 for end in bounds)), bounds)) + (
        f"{bounds[-1] + 1}+" if bounds else "1+",
    )


class RecoveryMetrics:
    """每个视频重置身份历史，分母由真值遮挡事件决定。"""

    def __init__(self, threshold: float, bins: tuple[int, ...], min_score: float) -> None:
        self.threshold = threshold
        self.bins = bins
        self.labels = _occlusion_labels(bins)
        self.min_score = min_score
        self.events = {label: (Counter(), Counter()) for label in ("all", *self.labels)}
        self.reset_sequence()
        self.calibration = {"clear": self._score_bins(), "partial": self._score_bins(), "reappeared": self._score_bins()}

    def reset_sequence(self) -> None:
        self.previous = {}
        self.pending = {}
        self.owners = {}

    @staticmethod
    def _score_bins() -> dict:
        return {
            "missed": 0,
            "counts": [0] * 5,
            "sum_iou_error": [0.0] * 5,
        }

    def observe(self, truth: GroundTruth, frame: FrameDetections, result: FrameTracks) -> np.ndarray:
        tracks = np.array([[*item.bbox_xyxy, 1, item.class_id] for item in result.tracks]).reshape(-1, 6)
        track_iou = _iou(truth.visible, tracks)
        gt_indices, track_indices = _assign_iou(track_iou, track_iou >= self.threshold)
        matches = {truth.ids[i]: result.tracks[j].track_id for i, j in zip(gt_indices, track_indices)}
        current_owners = {track_id: gt_id for gt_id, track_id in matches.items()}

        detections = frame.detections[frame.detections[:, 4] >= self.min_score]
        det_iou = _iou(truth.visible, detections)
        gt_indices, det_indices = _assign_iou(det_iou, det_iou >= self.threshold)
        detector_matches = dict(zip(gt_indices, det_indices))

        for index, gt_id in enumerate(truth.ids):
            gap = self.pending.get(gt_id)
            detection_index = detector_matches.get(index)
            self._record_detection(gt_id, gap is not None, truth.partial, index, detection_index, detections, det_iou)
            if gap is None:
                continue
            anchor, missing = gap
            current = matches.get(gt_id)
            if anchor is not None and current == anchor:
                outcome = "recovered"
            elif (
                (anchor in current_owners and current_owners[anchor] != gt_id)
                or (current in self.owners and self.owners[current] != gt_id)
            ):
                outcome = "wrong_reconnect"
            elif anchor is None:
                outcome = "missing_anchor"
            else:
                outcome = "unmatched" if current is None else "new_id"
            label = self.labels[bisect_left(self.bins, missing)]
            for key in ("all", label):
                all_events, with_detection = self.events[key]
                all_events[outcome] += 1
                if detection_index is not None:
                    with_detection[outcome] += 1

        pending = {}
        for gt_id in truth.occluded:
            if gt_id in self.pending:
                anchor, missing = self.pending[gt_id]
                pending[gt_id] = anchor, missing + 1
            elif gt_id in self.previous:
                pending[gt_id] = self.previous[gt_id], 1
        self.pending = pending
        self.previous = {gt_id: matches.get(gt_id) for gt_id in truth.ids}
        self.owners.update(current_owners)
        return track_iou

    def _record_detection(self, gt_id, reappeared, partial, gt_index, detection_index, detections, ious) -> None:
        category = "reappeared" if reappeared else "partial" if gt_id in partial else "clear"
        stats = self.calibration[category]
        if detection_index is None:
            stats["missed"] += 1
        else:
            bucket = min(4, int(detections[detection_index, 4] * 5))
            stats["counts"][bucket] += 1
            stats["sum_iou_error"][bucket] += float(1 - ious[gt_index, detection_index])

    def report(self) -> dict:
        return {
            label: {
                "all_events": _event_report(all_events),
                "with_detection": _event_report(detected),
                "detector_misses": sum(all_events.values()) - sum(detected.values()),
            }
            for label, (all_events, detected) in self.events.items()
        }

    def calibration_report(self) -> dict:
        return {
            label: {
                "missed": stats["missed"],
                "score_bins": [
                    {"min_score": i / 5, "max_score": (i + 1) / 5, "count": count,
                     "mean_iou_error": total / count if count else None}
                    for i, (count, total) in enumerate(zip(stats["counts"], stats["sum_iou_error"]))
                ],
            }
            for label, stats in self.calibration.items()
        }


def _event_report(events: Counter) -> dict:
    total = sum(events.values())
    return {
        "events": total,
        **{key: events[key] for key in ("recovered", "wrong_reconnect", "missing_anchor", "unmatched", "new_id")},
        "same_id_rate": events["recovered"] / total if total else None,
        "wrong_reconnect_rate": events["wrong_reconnect"] / total if total else None,
    }


class MOTSequence:
    """为 TrackEval 提供每序列连续索引和类别门控后的 IoU。"""

    def __init__(self) -> None:
        self.truth_ids = {}
        self.predicted_ids = {}
        self.gt = []
        self.predicted = []
        self.similarities = []

    def append(self, truth: GroundTruth, result: FrameTracks, ious: np.ndarray) -> None:
        self.gt.append(np.array([self.truth_ids.setdefault(i, len(self.truth_ids)) for i in truth.ids], dtype=int))
        self.predicted.append(np.array([
            self.predicted_ids.setdefault(item.track_id, len(self.predicted_ids)) for item in result.tracks
        ], dtype=int))
        self.similarities.append(ious)

    def data(self) -> dict:
        return {
            "gt_ids": self.gt, "tracker_ids": self.predicted, "similarity_scores": self.similarities,
            "num_timesteps": len(self.gt), "num_gt_ids": len(self.truth_ids), "num_tracker_ids": len(self.predicted_ids),
            "num_gt_dets": sum(map(len, self.gt)), "num_tracker_dets": sum(map(len, self.predicted)),
        }
