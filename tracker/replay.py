"""遮挡轨迹的有界滤波回放，只处理已成功重关联的轨迹。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .kalman import TrackingKalman


@dataclass(frozen=True)
class ORUConfig:
    max_missing: int = 6
    max_steps: int = 16
    noise_scale: float = 4.0

    def __post_init__(self) -> None:
        for field in ("max_missing", "max_steps"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"ORU {field} 必须是正整数")
        if not isfinite(self.noise_scale) or self.noise_scale < 1:
            raise ValueError("ORU 虚拟观测噪声倍率必须为有限数且不小于 1")


@dataclass(frozen=True)
class ReplaySnapshot:
    frame_id: int
    measurement: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def capture(
        cls, frame_id: int, measurement: np.ndarray, mean: np.ndarray, covariance: np.ndarray
    ) -> ReplaySnapshot:
        # 保留真实检测的 XYAH 观测，避免持有整帧的候选或批量滤波数组。
        return cls(frame_id, measurement.copy(), mean.copy(), covariance.copy())


def _valid_state(mean: np.ndarray, covariance: np.ndarray) -> bool:
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or (mean[2:4] <= 0).any():
        return False
    cx, cy, aspect, height = mean[:4]
    half_w, half_h = 0.5 * aspect * height, 0.5 * height
    if not (cx - half_w < cx + half_w and cy - half_h < cy + half_h):
        return False
    asymmetry = np.abs(covariance - covariance.T).max()
    if asymmetry > 1e-9 + 1e-7 * np.abs(covariance).max():
        return False
    np.linalg.cholesky(covariance)
    return True


def replay_prediction(
    kalman: TrackingKalman,
    snapshot: ReplaySnapshot,
    measurement: np.ndarray,
    missing: int,
    noise_scale: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, int]:
    """从实测后的快照推进至当前帧预测，不融合当前检测。"""
    mean, covariance = snapshot.mean.copy(), snapshot.covariance.copy()
    attempted = 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if not _valid_state(mean, covariance):
                return None, attempted
            step_delta = (measurement - snapshot.measurement) / (missing + 1)
            for step in range(1, missing + 1):
                attempted += 1
                mean, covariance = kalman.predict(mean, covariance)
                if (mean[2:4] <= 0).any():
                    return None, attempted
                virtual = snapshot.measurement + step * step_delta
                mean, covariance = kalman.update_virtual(mean, covariance, virtual, noise_scale)
                if not _valid_state(mean, covariance):
                    return None, attempted
            prior = kalman.predict(mean, covariance)
            return (prior if _valid_state(*prior) else None), attempted
    except (np.linalg.LinAlgError, FloatingPointError):
        return None, attempted
