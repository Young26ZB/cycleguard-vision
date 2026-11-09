"""固定帧率的检测缓存和逐帧轨迹契约。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Iterator, TextIO

import numpy as np


@dataclass(frozen=True)
class VideoSource:
    source_id: str
    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id 必须是非空字符串")
        if type(self.fps) not in (int, float) or not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps 必须是有限正数")
        if any(type(size) is not int or size <= 0 for size in (self.width, self.height)):
            raise ValueError("原图 width 和 height 必须是正整数")

    def cache_header(self) -> dict:
        return {"schema_version": 1, **asdict(self)}


@dataclass(frozen=True)
class FrameDetections:
    source: VideoSource
    frame_id: int
    timestamp_s: float
    # 原始顺序的 N x 6 数组。每行是 x1, y1, x2, y2, score, class_id。
    detections: np.ndarray

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id 必须是非负整数")
        if type(self.timestamp_s) not in (int, float) or not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s 必须是有限非负数")
        rows = np.asarray(self.detections)
        if rows.shape == (0,):
            rows = rows.reshape(0, 6)
        if rows.ndim != 2 or rows.shape[1] != 6 or rows.dtype.kind not in "iuf":
            raise ValueError("detections 必须是 N×6 数值数组")
        rows = rows.astype(np.float64, copy=False)
        if len(rows):
            if not np.isfinite(rows).all():
                raise ValueError("检测框、置信度和类别必须为有限数值")
            if (
                (rows[:, :2] < 0).any()
                or (rows[:, 2:4] <= rows[:, :2]).any()
                or (rows[:, 2] > self.source.width).any()
                or (rows[:, 3] > self.source.height).any()
            ):
                raise ValueError("检测框必须是原图范围内的正面积 xyxy")
            scores, classes = rows[:, 4], rows[:, 5]
            if ((scores < 0) | (scores > 1)).any():
                raise ValueError("score 必须在 [0, 1] 内")
            if ((classes < 0) | (classes != np.floor(classes))).any():
                raise ValueError("class_id 必须是非负整数")
        object.__setattr__(self, "detections", rows)

    def cache_record(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "detections": self.detections.tolist(),
        }


class FrameSequence:
    """检查序列时基，每个来源只需保留上一帧的元数据。"""

    def __init__(self) -> None:
        self.source: VideoSource | None = None
        self.frame_id = -1
        self.timestamp_s = -1.0

    def advance(self, frame: FrameDetections) -> bool:
        """接受一帧并返回是否切换来源。非法帧不推进序列。"""
        new_source = self.source is None or self.source.source_id != frame.source.source_id
        if not new_source and self.source != frame.source:
            raise ValueError("同一来源的帧率和尺寸必须保持一致")
        expected_id = 0 if new_source else self.frame_id + 1
        if frame.frame_id != expected_id:
            raise ValueError(f"frame_id 应为 {expected_id}，实际为 {frame.frame_id}，缓存必须连续")
        period = 1.0 / frame.source.fps
        tolerance = min(0.1 * period, 0.001)
        if (
            (not new_source and frame.timestamp_s <= self.timestamp_s)
            or abs(frame.timestamp_s - frame.frame_id / frame.source.fps) > tolerance
        ):
            raise ValueError("timestamp_s 必须递增并与固定帧率对齐")
        self.source = frame.source
        self.frame_id = frame.frame_id
        self.timestamp_s = frame.timestamp_s
        return new_source


def read_cache(stream: TextIO) -> Iterator[FrameDetections]:
    """逐行消费一个版本 1 的 JSONL 缓存。"""
    try:
        header = json.loads(next(stream))
        if not isinstance(header, dict) or type(header.get("schema_version")) is not int or header["schema_version"] != 1:
            raise ValueError("缺少支持的 schema_version 1")
        source = VideoSource(*(header[key] for key in ("source_id", "fps", "width", "height")))
    except (StopIteration, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"第 1 行不是有效的检测缓存头（需要 JSONL，带框视频不能作为缓存）：{exc}") from exc

    sequence = FrameSequence()
    for line_number, line in enumerate(stream, 2):
        try:
            row = json.loads(line)
            frame = FrameDetections(source, row["frame_id"], row["timestamp_s"], row["detections"])
            sequence.advance(frame)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"检测缓存第 {line_number} 行无效：{exc}") from exc
        yield frame
    if sequence.frame_id < 0:
        raise ValueError("检测缓存没有帧记录")


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]
    detection_index: int
    match_source: str | None = None


@dataclass(frozen=True)
class TrackPrediction:
    track_id: int
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]
    provenance: str = "prediction"


@dataclass(frozen=True)
class FrameStats:
    input_detections: int
    association_candidates: int
    truncated_candidates: int
    rejected_new_tracks: int
    live_tracks: int
    association_rows: int
    association_columns: int


@dataclass(frozen=True)
class AssociationStats:
    ocm_direction_tracks: int = 0
    ocr_calls: int = 0
    ocr_skipped: int = 0
    ocr_rows: int = 0
    ocr_columns: int = 0
    ocr_truncated_tracks: int = 0
    ocr_truncated_detections: int = 0
    ocr_matches: int = 0


@dataclass(frozen=True)
class ReplayStats:
    oru_requests: int = 0
    oru_replayed: int = 0
    oru_steps: int = 0
    oru_track_limit: int = 0
    oru_frame_limit: int = 0
    oru_missing_snapshot: int = 0
    oru_invalid_state: int = 0
    oru_fallbacks: int = 0


@dataclass(frozen=True)
class FrameTracks:
    source_id: str
    frame_id: int
    timestamp_s: float
    tracks: tuple[TrackObservation, ...]
    stats: FrameStats
    predictions: tuple[TrackPrediction, ...] = ()
    association: AssociationStats | None = None
    replay: ReplayStats | None = None

    def output_record(self) -> dict:
        tracks = [vars(track).copy() for track in self.tracks]
        if self.association is None:
            for track in tracks:
                del track["match_source"]
        record = {
            "source_id": self.source_id,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "tracks": tracks,
            "stats": vars(self.stats).copy(),
        }
        if self.predictions:
            record["predictions"] = [vars(prediction).copy() for prediction in self.predictions]
        if self.association is not None:
            record["association"] = vars(self.association).copy()
        if self.replay is not None:
            record["oru"] = vars(self.replay).copy()
        return record
