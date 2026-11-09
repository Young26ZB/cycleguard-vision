"""对主链路的独立卡尔曼测量进行批量校正。"""

from __future__ import annotations

import numpy as np

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


_MEASUREMENT_DIAGONAL = np.diag_indices(4)


class TrackingKalman(KalmanFilterXYAH):
    def update_batch(
        self, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量融合各轨迹已匹配的观测，返回校正后的均值和协方差。"""
        projected, innovation = self.project(means, covariances)
        return self._correct(means, covariances, measurements, projected, innovation)

    def update_virtual(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, noise_scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """用独立的保守观测噪声校正单条轨迹的虚拟观测。"""
        noise = np.maximum(self._measurement_variances(mean) * noise_scale, 1e-12)
        # XYAH 观测直接读取状态的前四维。
        innovation = covariance[:4, :4].copy()
        innovation[_MEASUREMENT_DIAGONAL] += noise
        return self._correct(mean, covariance, measurement, mean[:4], innovation)

    def _correct(self, means, covariances, measurements, projected, innovation):
        cross = covariances @ self._update_mat.T
        gain = np.linalg.solve(innovation, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
        residual = measurements - projected
        updated_means = means + (gain @ residual[..., None])[..., 0]
        updated_covariances = covariances - gain @ innovation @ gain.swapaxes(-1, -2)
        return updated_means, updated_covariances
