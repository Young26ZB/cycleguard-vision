"""固定步长的 XYAH 卡尔曼预测与一对一 IoU 跟踪。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_ioa

from .association import OCMConfig, OCRConfig, ObservationMotion, apply_momentum
from .kalman import TrackingKalman
from .nsa import NSAConfig, update_real_observations
from .records import (
    AssociationStats,
    FrameDetections,
    FrameSequence,
    FrameStats,
    FrameTracks,
    NSAStats,
    ReplayStats,
    TrackObservation,
    TrackPrediction,
    VideoSource,
)
from .replay import ORUConfig, ReplaySnapshot, replay_prediction


_EMPTY_REPLAY_STATS = ReplayStats()
_EMPTY_NSA_STATS = NSAStats()


@dataclass(frozen=True)
class TrackingConfig:
    min_hits: int = 3
    max_age: int = 15
    max_tracks: int = 128
    max_detections: int = 256
    min_score: float = 0.25
    new_track_score: float = 0.5
    min_iou: float = 0.3

    def __post_init__(self) -> None:
        for field in ("min_hits", "max_age", "max_tracks", "max_detections"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} 必须是正整数")
        if not 0 <= self.min_score <= self.new_track_score <= 1:
            raise ValueError("分数阈值须满足 0 <= min_score <= new_track_score <= 1")
        if not 0 < self.min_iou <= 1:
            raise ValueError("min_iou 须在 (0, 1] 内")


class TrackState(Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"


@dataclass
class _Track:
    track_id: int
    class_id: int
    state: TrackState
    mean: np.ndarray
    covariance: np.ndarray
    hits: int
    last_frame: int
    motion: ObservationMotion | None = None
    last_box: np.ndarray | None = None
    snapshot: ReplaySnapshot | None = None


def _xyah_to_boxes(means: np.ndarray, image_limit: np.ndarray) -> np.ndarray:
    half_size = means[:, 2:4].copy()
    half_size[:, 0] *= half_size[:, 1]
    half_size *= 0.5
    center = means[:, :2]
    return np.concatenate(
        (np.clip(center - half_size, 0, image_limit), np.clip(center + half_size, 0, image_limit)), axis=1
    )


def _assign_iou(ious: np.ndarray, valid: np.ndarray, *, max_cost: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """先最大化可行匹配数，再最大化总分。"""
    rows, columns = np.nonzero(valid)
    if not rows.size:
        return rows, columns
    # 通常每个框只有一个可行邻居，可直接跳过分配求解。
    row_degree = np.bincount(rows, minlength=valid.shape[0])
    col_degree = np.bincount(columns, minlength=valid.shape[1])
    if row_degree.max() == col_degree.max() == 1:
        return rows, columns

    eligible_rows = np.flatnonzero(row_degree)
    eligible_cols = np.flatnonzero(col_degree)
    transposed = eligible_rows.size > eligible_cols.size
    if transposed:
        ious, valid = ious.T, valid.T
        eligible_rows, eligible_cols = eligible_cols, eligible_rows
    row_count, col_count = len(eligible_rows), len(eligible_cols)
    selection = np.ix_(eligible_rows, eligible_cols)
    # 每行都有可用的虚拟列。弃配代价高于所有真边可能的总差值。
    costs = np.full((row_count, col_count + row_count), row_count * max_cost + 1.0)
    costs[:, :col_count] = np.where(valid[selection], 1.0 - ious[selection], np.inf)
    assigned_rows, assigned_cols = linear_sum_assignment(costs)
    matched = assigned_cols < col_count
    first = eligible_rows[assigned_rows[matched]]
    second = eligible_cols[assigned_cols[matched]]
    return (second, first) if transposed else (first, second)


def _gated_iou(boxes, classes, detections, min_iou):
    ious = bbox_ioa(boxes, detections[:, :4], iou=True, eps=0.0)
    return ious, (ious >= min_iou) & (classes[:, None] == detections[:, 5])


class Tracker:
    """每次 update 消费一帧，轨迹状态仅在当前视频内有效。"""

    def __init__(
        self, config: TrackingConfig | None = None, *, debug_predictions: bool = False,
        ocm: OCMConfig | None = None, ocr: OCRConfig | None = None, oru: ORUConfig | None = None,
        nsa: NSAConfig | None = None,
    ) -> None:
        self.config = config if config is not None else TrackingConfig()
        self.debug_predictions = debug_predictions
        self.ocm = ocm
        self.ocr = ocr
        self.oru = oru
        self.nsa = nsa
        self._kalman = TrackingKalman()
        self._sequence = FrameSequence()
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    @property
    def source(self) -> VideoSource | None:
        return self._sequence.source

    def _predict(self, tracks: list[_Track]) -> np.ndarray:
        if not tracks:
            return np.empty((0, 8))
        old_means = np.stack([track.mean for track in tracks])
        means, covariances = self._kalman.multi_predict(
            old_means, np.stack([track.covariance for track in tracks])
        )
        # 预测尺度穿过零时保留上一个有效尺度，并停止该维度的外推。
        invalid_size = means[:, 2:4] <= 0
        np.copyto(means[:, 2:4], old_means[:, 2:4], where=invalid_size)
        np.copyto(means[:, 6:8], 0.0, where=invalid_size)
        for track, mean, covariance in zip(tracks, means, covariances):
            track.mean = mean
            track.covariance = covariance
        return means

    def _observe(
        self, tracks: list[_Track], measurements: np.ndarray, scores: np.ndarray | None, frame_id: int,
    ) -> tuple[ReplayStats | None, np.ndarray | None, NSAStats | None]:
        means = np.stack([track.mean for track in tracks])
        covariances = np.stack([track.covariance for track in tracks])
        replay = self._replay_matches(tracks, measurements, frame_id, means, covariances) if self.oru else None
        accepted = nsa_stats = None
        if self.nsa is None:
            means, covariances = self._kalman.update_batch(means, covariances, measurements)
        else:
            means, covariances, accepted, nsa_stats = update_real_observations(
                self._kalman, means, covariances, measurements, scores, self.nsa
            )
        valid_sizes = (means[:, 2:4] > 0).all(axis=1)
        for index, (track, mean, covariance, measurement, usable) in enumerate(
            zip(tracks, means, covariances, measurements, valid_sizes)
        ):
            if accepted is not None and not accepted[index]:
                continue
            if not usable:
                mean, covariance = self._kalman.initiate(measurement)
            track.mean, track.covariance = mean, covariance
            track.hits += 1
            track.last_frame = frame_id
            if self.oru is not None:
                track.snapshot = ReplaySnapshot.capture(frame_id, measurement, mean, covariance)
            if track.motion is not None:
                track.motion.observe(frame_id, *measurement[:2], self.ocm)
            if track.hits >= self.config.min_hits:
                track.state = TrackState.ACTIVE
        return replay, accepted, nsa_stats

    def _replay_matches(
        self, tracks: list[_Track], measurements: np.ndarray, frame_id: int, means: np.ndarray, covariances: np.ndarray,
    ) -> ReplayStats:
        config = self.oru
        requests = sorted(
            (frame_id - track.last_frame - 1, track.track_id, index)
            for index, track in enumerate(tracks)
            if track.state is TrackState.LOST and frame_id - track.last_frame > 1
        )
        if not requests:
            return _EMPTY_REPLAY_STATS

        steps = replayed = track_limit = frame_limit = missing_snapshot = invalid = 0
        for missing, _, index in requests:
            track = tracks[index]
            if missing > config.max_missing:
                track_limit += 1
                continue
            snapshot = track.snapshot
            if snapshot is None or snapshot.frame_id != track.last_frame:
                missing_snapshot += 1
                continue
            if missing > config.max_steps - steps:
                frame_limit += 1
                continue
            prior, attempted = replay_prediction(
                self._kalman, snapshot, measurements[index], missing, config.noise_scale
            )
            steps += attempted
            if prior is None:
                invalid += 1
            else:
                means[index], covariances[index] = prior
                replayed += 1
        return ReplayStats(
            oru_requests=len(requests), oru_replayed=replayed, oru_steps=steps, oru_track_limit=track_limit,
            oru_frame_limit=frame_limit, oru_missing_snapshot=missing_snapshot, oru_invalid_state=invalid,
            oru_fallbacks=len(requests) - replayed,
        )

    def _recover(self, tracks, established, matches, available, candidates, frame_id, direction_tracks):
        """补一次剩余项的关联，原地登记成功的检测索引。"""
        config = self.ocr
        if not available.size:
            return (), available, AssociationStats(direction_tracks, ocr_skipped=1)
        window = min(config.max_age, self.config.max_age)
        eligible = [
            i for i in established
            if matches[i] < 0 and frame_id - tracks[i].last_frame <= window
        ]
        if not eligible:
            return (), available, AssociationStats(direction_tracks, ocr_skipped=1)
        if len(eligible) > config.max_tracks:
            eligible.sort(key=lambda i: (-tracks[i].last_frame, tracks[i].track_id))
        rows = np.asarray(eligible[:config.max_tracks], dtype=np.intp)
        columns = available[:config.max_detections]
        last_boxes = np.stack([tracks[i].last_box for i in rows])
        classes = np.array([tracks[i].class_id for i in rows])
        ious, valid = _gated_iou(last_boxes, classes, candidates[columns], config.min_iou)
        matched_rows, matched_cols = _assign_iou(ious, valid)
        recovered = rows[matched_rows]
        matches[recovered] = columns[matched_cols]
        keep = np.ones(len(available), dtype=bool)
        keep[matched_cols] = False
        return recovered, available[keep], AssociationStats(
            ocm_direction_tracks=direction_tracks, ocr_calls=1, ocr_rows=len(rows), ocr_columns=len(columns),
            ocr_truncated_tracks=len(eligible) - len(rows), ocr_truncated_detections=len(available) - len(columns),
            ocr_matches=len(recovered),
        )

    def update(self, frame: FrameDetections) -> FrameTracks:
        if self._sequence.advance(frame):
            self._tracks.clear()
            self._next_id = 1
        config = self.config
        self._tracks = {
            key: track for key, track in self._tracks.items()
            if frame.frame_id - track.last_frame <= config.max_age
        }
        tracks = list(self._tracks.values())
        image_limit = np.array([frame.source.width, frame.source.height])
        predicted_boxes = _xyah_to_boxes(self._predict(tracks), image_limit)

        rows = frame.detections
        eligible = np.flatnonzero(rows[:, 4] >= config.min_score)
        ordered = eligible[np.argsort(-rows[eligible, 4], kind="stable")]
        selected = ordered[:config.max_detections]
        candidates = rows[selected]
        sizes = candidates[:, 2:4] - candidates[:, :2]
        measurements = np.column_stack((
            candidates[:, :2] + sizes / 2, sizes[:, 0] / sizes[:, 1], sizes[:, 1]
        ))

        matched_candidates = np.full(len(tracks), -1, dtype=np.intp)
        available = np.arange(len(candidates))
        tentative = np.array([t.state is TrackState.TENTATIVE for t in tracks], dtype=bool)
        track_classes = np.array([track.class_id for track in tracks])
        observed = []
        predicted = []
        max_rows = max_columns = 0
        direction_tracks = 0

        established = np.flatnonzero(~tentative)
        for group in (established, np.flatnonzero(tentative)):
            if not group.size or not available.size:
                continue
            max_rows = max(max_rows, group.size)
            max_columns = max(max_columns, available.size)
            ious, valid = _gated_iou(predicted_boxes[group], track_classes[group], candidates[available], config.min_iou)
            used_directions = 0
            if self.ocm is not None and group.size + available.size > 2:
                motions = [
                    tracks[i].motion.direction if frame.frame_id - tracks[i].last_frame <= self.ocm.max_age else None
                    for i in group
                ]
                used_directions = apply_momentum(ious, valid, measurements[available, :2], motions)
                direction_tracks += used_directions
            matched_group, matched_available = _assign_iou(ious, valid, max_cost=2.0 if used_directions else 1.0)
            matched_candidates[group[matched_group]] = available[matched_available]
            keep = np.ones(len(available), dtype=bool)
            keep[matched_available] = False
            available = available[keep]

        association = None
        recovered = ()
        if self.ocr is not None:
            recovered, available, association = self._recover(
                tracks, established, matched_candidates, available, candidates, frame.frame_id, direction_tracks
            )
        elif self.ocm is not None:
            association = AssociationStats(direction_tracks)
        recovered = set(recovered) if len(recovered) else ()

        replay = _EMPTY_REPLAY_STATS if self.oru is not None else None
        nsa_stats = _EMPTY_NSA_STATS if self.nsa is not None else None
        real_updates = 0
        matched = matched_candidates >= 0
        if matched.any():
            track_indexes = np.flatnonzero(matched)
            matched_tracks = [tracks[i] for i in track_indexes]
            candidate_indexes = matched_candidates[track_indexes]
            matched_measurements = measurements[candidate_indexes]
            scores = candidates[candidate_indexes, 4] if self.nsa is not None else None
            replay, accepted, nsa_stats = self._observe(matched_tracks, matched_measurements, scores, frame.frame_id)
            if accepted is not None:
                matched[track_indexes[~accepted]] = False
            real_updates = len(track_indexes) - (nsa_stats.nsa_failed_updates if nsa_stats is not None else 0)
            for index, track, i in zip(track_indexes, matched_tracks, candidate_indexes):
                if not matched[index]:
                    continue
                if self.ocr is not None:
                    track.last_box = candidates[i, :4].copy()
                if track.state is TrackState.ACTIVE:
                    source = None
                    if association is not None:
                        source = "ocr" if index in recovered else "primary"
                    observed.append((track, candidates[i], int(selected[i]), source))

        for track_idx in np.flatnonzero(~matched):
            track = tracks[track_idx]
            failed = matched_candidates[track_idx] >= 0
            if track.state is TrackState.TENTATIVE or (failed and not (
                np.isfinite(track.mean).all() and np.isfinite(track.covariance).all()
            )):
                del self._tracks[track.track_id]
                continue
            track.state = TrackState.LOST
            if self.debug_predictions:
                box = predicted_boxes[track_idx]
                if (box[2:] > box[:2]).all():
                    predicted.append(TrackPrediction(track.track_id, track.class_id, tuple(box)))

        rejected = 0
        for candidate_idx in available:
            row = candidates[candidate_idx]
            if row[4] < config.new_track_score:
                break  # candidates remain in descending score order
            if len(self._tracks) == config.max_tracks:
                rejected += 1
                continue
            state = TrackState.ACTIVE if config.min_hits == 1 else TrackState.TENTATIVE
            mean, covariance = self._kalman.initiate(measurements[candidate_idx])
            track = _Track(self._next_id, int(row[5]), state, mean, covariance, 1, frame.frame_id)
            if self.oru is not None:
                track.snapshot = ReplaySnapshot.capture(frame.frame_id, measurements[candidate_idx], mean, covariance)
            if self.ocm is not None:
                track.motion = ObservationMotion(self.ocm)
                track.motion.observe(frame.frame_id, *measurements[candidate_idx, :2], self.ocm)
            if self.ocr is not None:
                track.last_box = row[:4].copy()
            self._tracks[track.track_id] = track
            self._next_id += 1
            if state is TrackState.ACTIVE:
                observed.append((track, row, int(selected[candidate_idx]), "new" if association is not None else None))

        return FrameTracks(
            frame.source.source_id, frame.frame_id, frame.timestamp_s,
            self._observations(observed, image_limit),
            FrameStats(len(rows), len(selected), len(eligible) - len(selected), rejected, len(self._tracks),
                       max_rows, max_columns),
            tuple(predicted),
            association,
            replay,
            nsa_stats,
            real_updates,
        )

    @staticmethod
    def _observations(
        observed: list[tuple[_Track, np.ndarray, int, str | None]], image_limit: np.ndarray
    ) -> tuple[TrackObservation, ...]:
        if not observed:
            return ()
        observed.sort(key=lambda item: item[0].track_id)
        boxes = _xyah_to_boxes(np.stack([track.mean for track, _, _, _ in observed]), image_limit)
        valid = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        return tuple(
            TrackObservation(track.track_id, track.class_id, tuple(box if usable else detection[:4]), index, source)
            for (track, detection, index, source), box, usable in zip(observed, boxes, valid)
        )
