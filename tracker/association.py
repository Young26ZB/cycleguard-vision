"""只用真实检测观测计算关联方向。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np


@dataclass(frozen=True)
class OCMConfig:
    mode: str = "fixed"
    weight: float = 0.2
    interval: int = 3
    max_age: int = 6
    min_displacement: float = 2.0
    history_size: int = 6

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "adaptive") or not 0 < self.weight <= 1:
            raise ValueError("OCM 需要 fixed 或 adaptive 模式，权重须在 (0, 1] 内")
        if type(self.interval) is not int or type(self.max_age) is not int or not 1 <= self.interval <= self.max_age:
            raise ValueError("OCM 观测间隔须在 1 和最大历史帧龄之间")
        if type(self.history_size) is not int or self.history_size < 2:
            raise ValueError("OCM 历史容量至少为 2")
        if not isfinite(self.min_displacement) or self.min_displacement <= 0:
            raise ValueError("OCM 最小位移须为有限正数")


@dataclass(frozen=True)
class OCRConfig:
    max_age: int = 6
    min_iou: float = 0.3
    max_tracks: int = 64
    max_detections: int = 128

    def __post_init__(self) -> None:
        for field in ("max_age", "max_tracks", "max_detections"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"OCR {field} 必须是正整数")
        if not 0 < self.min_iou <= 1:
            raise ValueError("OCR min_iou 须在 (0, 1] 内")


class ObservationMotion:
    def __init__(self, config: OCMConfig) -> None:
        self.history: deque[tuple[int, float, float]] = deque(maxlen=config.history_size)
        self.direction: tuple[float, float, float, float, float] | None = None

    def observe(self, frame_id: int, center_x: float, center_y: float, config: OCMConfig) -> None:
        self.history.append((frame_id, center_x, center_y))
        self.direction = None
        if len(self.history) < 2:
            return

        # 优先取达到目标间隔的最近观测，起步时允许较短的有效间隔。
        reference = len(self.history) - 1
        for index in range(reference - 1, -1, -1):
            age = frame_id - self.history[index][0]
            if age > config.max_age:
                break
            reference = index
            if age >= config.interval:
                break
        if reference == len(self.history) - 1:
            return

        _, x, y = self.history[reference]
        dx, dy = center_x - x, center_y - y
        distance = hypot(dx, dy)
        if not isfinite(distance) or distance < config.min_displacement:
            return
        weight = config.weight
        if config.mode == "adaptive":
            sum_x = sum_y = 0.0
            steps = 0
            prev_x, prev_y = x, y
            for index in range(reference + 1, len(self.history)):
                _, next_x, next_y = self.history[index]
                step_x, step_y = next_x - prev_x, next_y - prev_y
                length = hypot(step_x, step_y)
                if length:
                    sum_x += step_x / length
                    sum_y += step_y / length
                    steps += 1
                prev_x, prev_y = next_x, next_y
            coherence = min(1.0, hypot(sum_x, sum_y) / steps)
            weight *= min(1.0, distance / (2 * config.min_displacement)) * coherence
        self.direction = (x, y, dx / distance, dy / distance, weight)


def apply_momentum(
    scores: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    directions: list[tuple[float, float, float, float, float] | None],
) -> int:
    """原地调整可行且有竞争的边，保留原几何门控。"""
    if not any(direction is not None for direction in directions):
        return 0
    rows, cols = np.nonzero(valid)
    if not rows.size:
        return 0
    row_degree = np.bincount(rows, minlength=valid.shape[0])
    col_degree = np.bincount(cols, minlength=valid.shape[1])
    competitive = (row_degree[rows] > 1) | (col_degree[cols] > 1)
    if not competitive.any():
        return 0

    motion = np.asarray([direction if direction is not None else (0, 0, 0, 0, 0) for direction in directions])
    eligible = competitive & (motion[rows, 4] > 0)
    if not eligible.any():
        return 0
    rows, cols = rows[eligible], cols[eligible]
    displacement = centers[cols] - motion[rows, :2]
    norm = np.hypot(displacement[:, 0], displacement[:, 1])
    usable = (norm > 0) & np.isfinite(norm)
    rows, cols = rows[usable], cols[usable]
    if not rows.size:
        return 0
    unit = displacement[usable] / norm[usable, None]
    cosine = np.einsum("ij,ij->i", unit, motion[rows, 2:4]).clip(-1, 1)
    scores[rows, cols] -= motion[rows, 4] * (1 - cosine) * 0.5
    return int(np.count_nonzero(np.bincount(rows)))
