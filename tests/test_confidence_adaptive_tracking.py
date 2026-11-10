"""置信度自适应真实观测更新的非训练检查。"""

import unittest
from unittest.mock import patch

import numpy as np

from tracker.association import OCMConfig
from tracker.core import Tracker, TrackingConfig
from tracker.kalman import TrackingKalman
from tracker.nsa import NSAConfig, update_real_observations
from tracker.records import FrameDetections, VideoSource
from tracker.replay import ORUConfig


class ConfidenceUpdateTests(unittest.TestCase):
    source = VideoSource("nsa-test", 30, 800, 480)

    def frame(self, index, *boxes):
        rows = [[x - 22, 100, x + 22, 160, score, 0] for x, score in boxes]
        return FrameDetections(self.source, index, index / 30, rows)

    def test_scores_change_gain_with_positive_floor(self):
        kalman = TrackingKalman()
        measure = np.array([100.0, 130.0, 0.75, 60.0])
        mean, cov = kalman.initiate(measure)
        sample = measure + [12, 0, 0, 0]
        updated, covariances, valid, stats = update_real_observations(
            kalman, np.stack([mean] * 4), np.stack([cov] * 4), np.stack([sample] * 4),
            np.array([0.25, 0.75, 0.99, 1.0]), NSAConfig(),
        )
        self.assertTrue(valid.all())
        self.assertEqual(stats.nsa_fallbacks, 0)
        self.assertLess(updated[0, 0], updated[1, 0])
        self.assertLess(updated[1, 0], updated[2, 0])
        np.testing.assert_array_equal(updated[2], updated[3])
        np.linalg.cholesky(covariances)

    def test_only_bad_adaptive_row_is_retried(self):
        tracker = Tracker(TrackingConfig(min_hits=1), nsa=NSAConfig())
        tracker.update(self.frame(0, (100, 0.9), (300, 0.9)))
        update = tracker._kalman.update_batch
        calls = []

        def corrupt(means, covariances, measurements, **kwargs):
            calls.append((len(means), bool(kwargs)))
            output, covs = update(means, covariances, measurements, **kwargs)
            if kwargs:
                covs[0, 0, 1] = covs[0, 1, 0] = 1e8
            return output, covs

        with patch.object(tracker._kalman, "update_batch", side_effect=corrupt):
            result = tracker.update(self.frame(1, (105, 0.8), (305, 0.9)))
        self.assertEqual(calls, [(2, True), (1, False)])
        self.assertEqual([track.track_id for track in result.tracks], [1, 2])
        self.assertEqual(result.real_updates, 2)
        self.assertEqual((result.nsa.nsa_fallbacks, result.nsa.nsa_failed_updates), (1, 0))
        self.assertEqual([track.hits for track in tracker._tracks.values()], [2, 2])

    def test_reappearance_separates_virtual_noise_and_real_confidence(self):
        tracker = Tracker(TrackingConfig(min_hits=1), oru=ORUConfig(), nsa=NSAConfig())
        tracker.update(self.frame(0, (100, 0.9)))
        tracker.update(self.frame(1))
        with patch.object(tracker._kalman, "update_virtual", wraps=tracker._kalman.update_virtual) as virtual, \
             patch.object(tracker._kalman, "update_batch", wraps=tracker._kalman.update_batch) as real:
            result = tracker.update(self.frame(2, (110, 1.0)))
        virtual.assert_called_once()
        self.assertEqual(virtual.call_args.args[3], ORUConfig.noise_scale)
        real.assert_called_once()
        np.testing.assert_array_equal(real.call_args.kwargs["noise_scale"], [NSAConfig.min_scale])
        self.assertEqual(result.real_updates, 1)
        self.assertEqual(result.tracks[0].track_id, 1)

    def test_double_failure_keeps_normal_prior_and_real_history(self):
        tracker = Tracker(TrackingConfig(min_hits=1), ocm=OCMConfig(), oru=ORUConfig(), nsa=NSAConfig())
        tracker.update(self.frame(0, (100, 0.9)))
        tracker.update(self.frame(1))
        track = tracker._tracks[1]
        prior, covariance = tracker._kalman.predict(track.mean, track.covariance)
        with patch.object(tracker._kalman, "update_batch", side_effect=np.linalg.LinAlgError) as correct:
            result = tracker.update(self.frame(2, (110, 1.0)))
        self.assertEqual(correct.call_count, 2)
        self.assertEqual(result.real_updates, 0)
        self.assertEqual(result.nsa.nsa_failed_updates, 1)
        self.assertEqual(result.tracks, ())
        self.assertEqual(track.last_frame, 0)
        self.assertEqual(track.snapshot.frame_id, 0)
        self.assertEqual(track.motion.history[-1][0], 0)
        np.testing.assert_allclose(track.mean, prior)
        np.testing.assert_allclose(track.covariance, covariance)

    def test_disabled_keeps_original_call_and_output(self):
        tracker = Tracker(TrackingConfig(min_hits=1))
        tracker.update(self.frame(0, (100, 0.9)))
        with patch.object(tracker._kalman, "update_batch", wraps=tracker._kalman.update_batch) as real:
            result = tracker.update(self.frame(1, (105, 0.8)))
        real.assert_called_once()
        self.assertFalse(real.call_args.kwargs)
        self.assertNotIn("nsa", result.output_record())
        self.assertEqual(result.real_updates, 1)
