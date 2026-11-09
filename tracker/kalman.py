"""对主链路的独立卡尔曼测量进行批量校正。"""

from __future__ import annotations

import numpy as np

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


class TrackingKalman(KalmanFilterXYAH):
    def update_batch(
        self, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量融合各轨迹已匹配的观测，返回校正后的均值和协方差。"""
        projected, innovation = self.project(means, covariances)
        cross = covariances @ self._update_mat.T
        gain = np.linalg.solve(innovation, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
        residual = measurements - projected
        updated_means = means + (gain @ residual[..., None])[..., 0]
        updated_covariances = covariances - gain @ innovation @ gain.swapaxes(-1, -2)
        return updated_means, updated_covariances
