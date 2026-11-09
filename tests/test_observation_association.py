"""观测中心关联的非训练合成检查。"""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tracker.association import OCMConfig, OCRConfig, ObservationMotion, apply_momentum
from tracker.benchmark import association_benchmark
from tracker.core import Tracker, TrackingConfig, _assign_iou
from tracker.records import FrameDetections, VideoSource, read_cache
from tracker.track_ebike import write_tracks


class ObservationAssociationTests(unittest.TestCase):
    source = VideoSource("association-test", 30, 1024, 480)

    @staticmethod
    def box(x, score=0.9, cls=0):
        return [x - 22, 100, x + 22, 160, score, cls]

    def frame(self, index, *rows):
        return FrameDetections(self.source, index, index / 30, rows)

    @staticmethod
    def drift_prediction(tracker):
        predict = tracker._predict

        def displaced(tracks):
            means = predict(tracks).copy()
            means[:, 0] += 600
            return means

        return patch.object(tracker, "_predict", side_effect=displaced)

    def test_direction_breaks_equal_iou_without_opening_ineligible_edges(self):
        history = ObservationMotion(OCMConfig())
        history.observe(0, 100, 130, OCMConfig())
        history.observe(1, 104, 130, OCMConfig())
        scores = np.array([[0.5, 0.5, 0.99, 0.5]])
        valid = np.array([[True, True, False, True]])
        centers = np.array([[90, 130], [110, 130], [130, 130], [100, 130]])
        self.assertEqual(apply_momentum(scores, valid, centers, [history.direction]), 1)
        self.assertTrue(np.isfinite(scores).all())
        self.assertLess(scores[0, 0], scores[0, 1])
        self.assertEqual(scores[0, 2], 0.99)
        self.assertEqual(scores[0, 3], 0.5)  # candidate at reference has no angle
        # Give the undefined direction a slightly lower IoU so the forward candidate wins.
        scores[0, 3] -= 0.01
        self.assertEqual(_assign_iou(scores, valid, max_cost=2)[1].tolist(), [1])

    def test_ocm_uses_real_centers_before_primary_assignment(self):
        config = TrackingConfig(min_hits=1)
        tracker = Tracker(config, ocm=OCMConfig())
        tracker.update(self.frame(0, self.box(100)))
        tracker.update(self.frame(1, self.box(104)))
        mean = tracker._tracks[1].mean.copy()
        mean[0] = 100
        with patch.object(tracker, "_predict", return_value=mean[None, :]):
            result = tracker.update(self.frame(2, self.box(90), self.box(110), self.box(130, cls=1)))
        self.assertEqual((result.tracks[0].track_id, result.tracks[0].detection_index), (1, 1))
        self.assertEqual(result.tracks[0].match_source, "primary")
        self.assertEqual(result.association.ocm_direction_tracks, 1)
        self.assertEqual(tracker._tracks[1].motion.history[-1], (2, 110, 130))
        tracker.update(self.frame(3))
        self.assertEqual(tracker._tracks[1].motion.history[-1][0], 2)

    def test_stationary_stale_and_adaptive_direction_history(self):
        config = OCMConfig(interval=3, max_age=4, min_displacement=2, history_size=4)
        motion = ObservationMotion(config)
        for frame in range(10):
            motion.observe(frame, 50, 50, config)
        self.assertIsNone(motion.direction)
        self.assertEqual([point[0] for point in motion.history], [6, 7, 8, 9])
        motion.observe(14, 55, 50, config)
        self.assertIsNone(motion.direction)

        config = OCMConfig(mode="adaptive", interval=3)
        stable, turning = ObservationMotion(config), ObservationMotion(config)
        for index, (linear, zigzag) in enumerate(zip([100, 102, 104, 106], [100, 105, 102, 106])):
            stable.observe(index, linear, 0, config)
            turning.observe(index, zigzag, 0, config)
        self.assertAlmostEqual(stable.direction[-1], config.weight)
        self.assertGreater(stable.direction[-1], turning.direction[-1])

    def test_ocr_recovers_drifted_id_and_preserves_primary_match(self):
        config = TrackingConfig(min_hits=1)
        normal, recovering = Tracker(config), Tracker(config, ocr=OCRConfig())
        for index in range(10):
            rows = (self.box(100 + 16 * index), self.box(500)) if index < 6 else (self.box(500),)
            for tracker in (normal, recovering):
                tracker.update(self.frame(index, *rows))
        frame = self.frame(10, self.box(180), self.box(500))
        self.assertNotIn(1, [item.track_id for item in normal.update(frame).tracks])
        result = recovering.update(frame)
        self.assertEqual(
            [(item.track_id, item.detection_index, item.match_source) for item in result.tracks],
            [(1, 0, "ocr"), (2, 1, "primary")],
        )
        self.assertEqual((result.association.ocr_rows, result.association.ocr_columns), (1, 1))
        self.assertEqual(result.association.ocr_matches, 1)

    def test_ocr_window_and_tentative_exclusion(self):
        cases = (
            (TrackingConfig(min_hits=2), OCRConfig(), 1),  # tentative
            (TrackingConfig(min_hits=1), OCRConfig(max_age=1), 2),  # OCR age limit
            (TrackingConfig(min_hits=1, max_age=1), OCRConfig(max_age=6), 2),  # expired by main tracker
        )
        for config, recovery, index in cases:
            with self.subTest(config=config, recovery=recovery):
                tracker = Tracker(config, ocr=recovery)
                tracker.update(self.frame(0, self.box(100)))
                if index == 2:
                    tracker.update(self.frame(1))
                with self.drift_prediction(tracker):
                    result = tracker.update(self.frame(index, self.box(100)))
                self.assertEqual((result.association.ocr_calls, result.association.ocr_skipped), (0, 1))
                self.assertNotIn(1, [item.track_id for item in result.tracks])

    def test_ocr_bounds_use_recency_and_score_and_do_not_create_a_second_pass(self):
        tracker = Tracker(TrackingConfig(min_hits=1), ocr=OCRConfig(max_tracks=1, max_detections=1))
        tracker.update(self.frame(0, self.box(100), self.box(300)))
        tracker.update(self.frame(1, self.box(300)))
        with self.drift_prediction(tracker):
            result = tracker.update(self.frame(2, self.box(100, score=0.8), self.box(300, score=0.9)))
        stats = result.association
        self.assertEqual((stats.ocr_calls, stats.ocr_rows, stats.ocr_columns), (1, 1, 1))
        self.assertEqual((stats.ocr_truncated_tracks, stats.ocr_truncated_detections), (1, 1))
        self.assertEqual(
            [(item.track_id, item.detection_index) for item in result.tracks if item.match_source == "ocr"], [(2, 1)]
        )
        result = tracker.update(self.frame(3))
        self.assertEqual((result.association.ocr_calls, result.association.ocr_skipped), (0, 1))

    def test_disabled_keeps_output_schema_and_enabled_mode_keeps_original_detection_indices(self):
        path = Path(__file__).resolve().parents[1] / "tracker" / "example_detections.jsonl"
        for kwargs in ({}, {"ocm": OCMConfig()}, {"ocr": OCRConfig()}):
            with self.subTest(kwargs=kwargs), path.open(encoding="utf-8") as stream:
                output = io.StringIO()
                write_tracks(read_cache(stream), Tracker(**kwargs), output)
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual("association" in records[-1], bool(kwargs))
                self.assertEqual("match_source" in records[-1]["tracks"][0], bool(kwargs))
                self.assertEqual(records[-1]["tracks"][0]["detection_index"], 0)
                self.assertEqual(records[-1]["tracks"][0]["track_id"], 1)

    def test_benchmark_exercises_actual_ocr_and_reports_reappearance_subset(self):
        report = association_benchmark(TrackingConfig(max_tracks=4), frames=12, warmup=0, repeat=1, fps=30)
        case = report["cases"]["dense_misses"]["runs"]
        self.assertEqual(len(case), 6)
        self.assertEqual(case["off"][0]["reappearance_samples"], 1)
        self.assertNotIn("ocr_calls", case["off"][0]["workload_totals"])
        for mode in ("ocr", "ocm_fixed_ocr", "ocm_adaptive_ocr"):
            run = case[mode][0]
            self.assertGreater(run["workload_totals"]["ocr_matches"], 0)
            self.assertEqual(run["reappearance_samples"], 1)
            self.assertIn("p99", run["reappearance_latency_ms"])
            self.assertIsNone(run["end_to_end_latency_ms"])
