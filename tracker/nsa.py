"""真实观测的置信度噪声缩放及逐轨回退。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .records import NSAStats

if TYPE_CHECKING:
    from .kalman import TrackingKalman


@dataclass(frozen=True)
class NSAConfig:
    min_scale: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.min_scale <= 1:
            raise ValueError("NSA 最小噪声倍率须在 (0, 1] 内")


def _valid_updates(means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    valid = np.isfinite(means).all(axis=1) & np.isfinite(covariances).all(axis=(1, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        half_sizes = means[:, 2:4].copy() * 0.5
        half_sizes[:, 0] *= means[:, 3]
        low, high = means[:, :2] - half_sizes, means[:, :2] + half_sizes
        valid &= np.isfinite(low).all(axis=1) & np.isfinite(high).all(axis=1) & (high > low).all(axis=1)
        symmetry_error = np.abs(covariances - covariances.swapaxes(-1, -2)).max(axis=(1, 2))
        valid &= symmetry_error <= 1e-9 + 1e-7 * np.abs(covariances).max(axis=(1, 2))
    if valid.any():
        indices = np.flatnonzero(valid)
        try:
            np.linalg.cholesky(covariances if valid.all() else covariances[indices])
        except np.linalg.LinAlgError:
            for index in indices:
                try:
                    np.linalg.cholesky(covariances[index])
                except np.linalg.LinAlgError:
                    valid[index] = False
    return valid


def _attempt(kalman, means, covariances, measurements, scales=None):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if scales is None:
                return kalman.update_batch(means, covariances, measurements)
            return kalman.update_batch(means, covariances, measurements, noise_scale=scales)
    except (np.linalg.LinAlgError, FloatingPointError):
        return None


def update_real_observations(
    kalman: TrackingKalman, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray,
    scores: np.ndarray, config: NSAConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NSAStats]:
    scales = np.maximum(config.min_scale, 1.0 - scores)
    result = _attempt(kalman, means, covariances, measurements, scales)
    if result is None:
        updated_means, updated_covariances = means.copy(), covariances.copy()
        valid = np.zeros(len(means), dtype=bool)
    else:
        updated_means, updated_covariances = result
        valid = _valid_updates(updated_means, updated_covariances)
    failures = np.flatnonzero(~valid)
    if failures.size:
        retry = _attempt(kalman, means[failures], covariances[failures], measurements[failures])
        if retry is not None:
            updated_means[failures], updated_covariances[failures] = retry
            valid[failures] = _valid_updates(*retry)
        elif failures.size > 1:
            # 批量求解异常时隔离坏矩阵，让其他轨迹完成各自的普通更新。
            for index in failures:
                row = slice(index, index + 1)
                retry = _attempt(kalman, means[row], covariances[row], measurements[row])
                if retry is not None:
                    updated_means[row], updated_covariances[row] = retry
                    valid[row] = _valid_updates(*retry)
    return updated_means, updated_covariances, valid, NSAStats(int(failures.size), int((~valid).sum()))
