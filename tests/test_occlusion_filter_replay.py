"""ORU 回放、工作量上限和回退的非训练合成检查。"""

import json
import unittest
from unittest.mock import patch

import numpy as np

from tracker.association import OCMConfig, OCRConfig
from tracker.benchmark import oru_benchmark
from tracker.core import Tracker, TrackingConfig
from tracker.kalman import TrackingKalman
from tracker.records import FrameDetections, VideoSource
from tracker.replay import ORUConfig


class OcclusionReplayTests(unittest.TestCase):
    source = VideoSource("replay-test", 30, 1024, 480)

    @staticmethod
    def box(x):
        return [x - 22, 100, x + 22, 160, 0.9, 0]

    def frame(self, index, *centers):
        return FrameDetections(self.source, index, index / 30, [self.box(x) for x in centers])

    def test_snapshots_only_when_enabled_and_replaced_only_by_real_updates(self):
        for config in (None, ORUConfig()):
            with self.subTest(config=config):
                tracker = Tracker(TrackingConfig(min_hits=1), oru=config)
                first = tracker.update(self.frame(0, 100))
                track = tracker._tracks[1]
                anchor = track.snapshot
                self.assertEqual("oru" in first.output_record(), config is not None)
                second = tracker.update(self.frame(1, 102))
                self.assertEqual(first.tracks[0].track_id, second.tracks[0].track_id)
                if config is None:
                    self.assertIsNone(anchor)
                    self.assertIsNone(track.snapshot)
                    self.assertIsNone(second.replay)
                    continue
                self.assertIsNot(track.snapshot, anchor)
                self.assertEqual(anchor.frame_id, 0)
                self.assertEqual(track.snapshot.frame_id, 1)
                self.assertEqual(track.snapshot.measurement[0], 102)
                self.assertFalse(np.shares_memory(track.snapshot.mean, track.mean))
                self.assertFalse(np.shares_memory(track.snapshot.covariance, track.covariance))
                latest = track.snapshot
                self.assertEqual(tracker.update(self.frame(2)).replay.oru_requests, 0)
                self.assertIs(track.snapshot, latest)

    def test_three_virtual_observations_one_real_update_and_no_history_rewrite(self):
        tracker = Tracker(TrackingConfig(min_hits=1), ocm=OCMConfig(), oru=ORUConfig())
        for index in range(21):
            tracker.update(self.frame(index, 100))
        published = [tracker.update(self.frame(index)) for index in (21, 22, 23)]
        records = [json.dumps(result.output_record()) for result in published]
        with patch.object(tracker._kalman, "update_virtual", wraps=tracker._kalman.update_virtual) as virtual, \
             patch.object(tracker._kalman, "update_batch", wraps=tracker._kalman.update_batch) as real:
            current = tracker.update(self.frame(24, 116))
        self.assertEqual([call.args[2][0] for call in virtual.call_args_list], [104, 108, 112])
        self.assertTrue(all(call.args[3] == 4.0 for call in virtual.call_args_list))
        real.assert_called_once()
        np.testing.assert_allclose(real.call_args.args[2][0], [116, 130, 44 / 60, 60])
        self.assertEqual((current.replay.oru_requests, current.replay.oru_steps, current.replay.oru_replayed), (1, 3, 1))
        self.assertEqual((current.tracks[0].track_id, current.tracks[0].detection_index), (1, 0))
        self.assertEqual(current.tracks[0].match_source, "primary")
        self.assertEqual(tracker._tracks[1].motion.history[-1], (24, 116, 130))
        self.assertFalse({21, 22, 23} & {point[0] for point in tracker._tracks[1].motion.history})
        self.assertEqual([json.dumps(result.output_record()) for result in published], records)
        next_frame = tracker.update(self.frame(25, 120))
        self.assertEqual(next_frame.tracks[0].track_id, 1)
        self.assertEqual(next_frame.replay.oru_requests, 0)
        self.assertTrue(np.isfinite(tracker._tracks[1].covariance).all())

    def test_ocr_recovery_uses_replayed_prior_for_next_frame(self):
        config = TrackingConfig(min_hits=1)
        ordinary = Tracker(config, ocr=OCRConfig())
        replay = Tracker(config, ocr=OCRConfig(), oru=ORUConfig())
        for index in range(10):
            frame = self.frame(index, 100 + 16 * index) if index < 6 else self.frame(index)
            ordinary.update(frame)
            replay.update(frame)
        snapshot = replay._tracks[1].snapshot
        kalman = TrackingKalman()
        mean, covariance = snapshot.mean, snapshot.covariance
        for _ in range(4):
            mean, covariance = kalman.predict(mean, covariance)
            mean, covariance = kalman.update_virtual(mean, covariance, snapshot.measurement, 4.0)
        mean, covariance = kalman.predict(mean, covariance)
        mean, covariance = kalman.update(mean, covariance, snapshot.measurement)
        expected_next, _ = kalman.predict(mean, covariance)
        frame = self.frame(10, 180)
        ordinary_result, result = ordinary.update(frame), replay.update(frame)
        self.assertEqual(result.tracks[0].match_source, "ocr")
        self.assertEqual(result.tracks[0].track_id, ordinary_result.tracks[0].track_id)
        self.assertEqual(result.replay.oru_steps, 4)
        with patch.object(replay._kalman, "update_batch", wraps=replay._kalman.update_batch) as update:
            following = replay.update(self.frame(11, 180))
        np.testing.assert_allclose(update.call_args.args[0][0], expected_next, atol=1e-10)
        self.assertEqual(following.tracks[0].track_id, result.tracks[0].track_id)

    def test_prioritize_shorter_gaps_then_lower_ids_and_do_not_queue_work(self):
        for budget, expected in ((2, [300, 500]), (1, [300])):
            with self.subTest(budget=budget):
                tracker = Tracker(TrackingConfig(min_hits=1), oru=ORUConfig(max_steps=budget))
                tracker.update(self.frame(0, 100, 300, 500))
                tracker.update(self.frame(1, 300, 500))
                tracker.update(self.frame(2))
                with patch.object(tracker._kalman, "update_virtual", wraps=tracker._kalman.update_virtual) as virtual:
                    result = tracker.update(self.frame(3, 100, 300, 500))
                self.assertEqual([call.args[2][0] for call in virtual.call_args_list], expected)
                self.assertEqual([item.track_id for item in result.tracks], [1, 2, 3])
                self.assertEqual(result.replay.oru_frame_limit, 3 - budget)
                self.assertEqual(result.replay.oru_fallbacks, 3 - budget)
                self.assertEqual(result.replay.oru_steps, budget)
                self.assertEqual(tracker.update(self.frame(4, 100, 300, 500)).replay.oru_requests, 0)

    def test_failed_steps_consume_budget_and_preserve_normal_update(self):
        config = TrackingConfig(min_hits=1)
        ordinary, replay = Tracker(config), Tracker(config, oru=ORUConfig(max_steps=3))
        for frame in (self.frame(0, 100, 300), self.frame(1), self.frame(2)):
            ordinary.update(frame)
            replay.update(frame)
        update_virtual = replay._kalman.update_virtual
        call_count = 0

        def fail_second_step(mean, covariance, measurement, scale):
            nonlocal call_count
            call_count += 1
            mean, covariance = update_virtual(mean, covariance, measurement, scale)
            if call_count == 2:
                covariance[0, 1] = covariance[1, 0] = 1e6
            return mean, covariance

        frame = self.frame(3, 110, 310)
        baseline = ordinary.update(frame)
        with patch.object(replay._kalman, "update_virtual", side_effect=fail_second_step):
            result = replay.update(frame)
        self.assertEqual(result.tracks, baseline.tracks)
        for track_id in (1, 2):
            np.testing.assert_array_equal(replay._tracks[track_id].mean, ordinary._tracks[track_id].mean)
            np.testing.assert_array_equal(replay._tracks[track_id].covariance, ordinary._tracks[track_id].covariance)
        self.assertEqual(result.replay.oru_steps, 2)
        self.assertEqual(result.replay.oru_invalid_state, 1)
        self.assertEqual(result.replay.oru_frame_limit, 1)
        self.assertEqual(result.replay.oru_fallbacks, 2)

    def test_per_track_limit_and_missing_snapshot_fall_back(self):
        tracker = Tracker(TrackingConfig(min_hits=1), oru=ORUConfig(max_missing=1))
        tracker.update(self.frame(0, 100, 300))
        tracker.update(self.frame(1, 300))
        tracker.update(self.frame(2))
        tracker._tracks[2].snapshot = None
        result = tracker.update(self.frame(3, 100, 300))
        self.assertEqual(result.replay.oru_track_limit, 1)
        self.assertEqual(result.replay.oru_missing_snapshot, 1)
        self.assertEqual(result.replay.oru_steps, 0)
        self.assertEqual([item.track_id for item in result.tracks], [1, 2])
        self.assertEqual([tracker._tracks[i].snapshot.frame_id for i in (1, 2)], [3, 3])

    def test_virtual_noise_is_independent_and_weaker_than_real_update(self):
        kalman = TrackingKalman()
        initial = np.array([100, 130, 0.75, 60.0])
        mean, covariance = kalman.initiate(initial)
        measurement = initial + np.array([12, 0, 0, 0])
        real_mean, _ = kalman.update(mean, covariance, measurement)
        virtual_mean, _ = kalman.update_virtual(mean, covariance, measurement, 4.0)
        self.assertLess(mean[0], virtual_mean[0])
        self.assertLess(virtual_mean[0], real_mean[0])

    def test_scalar_prediction_matches_existing_batch_prediction(self):
        kalman = TrackingKalman()
        mean, covariance = kalman.initiate(np.array([100.0, 130.0, 0.75, 60.0]))
        mean[4:] = [6, -2, 0.01, 0.5]
        covariance[0, 4] = covariance[4, 0] = 1
        scalar = kalman.predict(mean, covariance)
        batch = kalman.multi_predict(mean[None, :], covariance[None, :, :])
        for expected, actual in zip(scalar, batch):
            np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)

    def test_benchmark_records_actual_replay_and_synchronized_subset(self):
        report = oru_benchmark(
            TrackingConfig(max_tracks=4), frames=12, warmup=0, repeat=1, fps=30, replay=ORUConfig(max_steps=4)
        )
        off, on = (report["runs"][mode][0] for mode in ("off", "on"))
        self.assertEqual(off["reappearance_samples"], on["reappearance_samples"])
        self.assertEqual(on["reappearance_samples"], 1)
        self.assertEqual(off["workload_totals"]["oru_steps"], 0)
        self.assertEqual(on["workload_peaks"]["oru_steps"], 4)
        self.assertGreater(on["workload_totals"]["oru_frame_limit"], 0)
        self.assertEqual(on["reappearance_replay_steps"], 4)
        self.assertIsNone(on["end_to_end_latency_ms"])
