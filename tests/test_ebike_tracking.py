"""电动车跟踪主链路的合成数据检查，无训练或权重。"""

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import cv2
import numpy as np
import torch

from tracker.benchmark import benchmark
from tracker.core import Tracker, TrackingConfig, _assign_iou
from tracker.kalman import TrackingKalman
from tracker.records import FrameDetections, VideoSource, read_cache
from tracker.track_ebike import main, write_tracks
from tracker.video import detections_from_result, video_detections


EXAMPLE = Path(__file__).resolve().parents[1] / "tracker" / "example_detections.jsonl"


def example_cache():
    return [json.loads(line) for line in EXAMPLE.read_text(encoding="utf-8").splitlines()]


def read_rows(rows):
    return read_cache(io.StringIO("\n".join(map(json.dumps, rows))))


class DetectionCacheTests(unittest.TestCase):
    def test_example_and_empty_frame(self):
        with EXAMPLE.open(encoding="utf-8") as stream:
            frames = list(read_cache(stream))
        self.assertEqual([frame.frame_id for frame in frames], list(range(5)))
        self.assertEqual(frames[3].detections.shape, (0, 6))
        self.assertEqual(frames[0].source.cache_header(), example_cache()[0])

    def test_invalid_frame_sequence_is_rejected_at_its_line(self):
        for invalid in ({"frame_id": 0}, {"frame_id": 2}, {"timestamp_s": 0.0}, {"timestamp_s": 0.06}):
            with self.subTest(invalid=invalid):
                rows = example_cache()
                rows[2].update(invalid)
                frames = read_rows(rows)
                self.assertEqual(next(frames).frame_id, 0)
                with self.assertRaisesRegex(ValueError, "第 3 行"):
                    next(frames)

    def test_invalid_boxes_and_scores(self):
        for index, value in ((0, 200), (2, 641), (1, -1), (3, float("nan")), (4, 1.1), (4, -0.1), (5, 0.5)):
            with self.subTest(index=index, value=value):
                rows = example_cache()
                rows[1]["detections"][0][index] = value
                with self.assertRaisesRegex(ValueError, "第 2 行"):
                    next(read_rows(rows))

    def test_invalid_metadata(self):
        for value in (0, float("inf"), "25"):
            with self.subTest(fps=value):
                rows = example_cache()
                rows[0]["fps"] = value
                with self.assertRaisesRegex(ValueError, "第 1 行"):
                    next(read_rows(rows))

    def test_rendered_video_or_empty_cache_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSONL"):
            next(read_cache(io.StringIO("0000ftypisom")))
        with self.assertRaisesRegex(ValueError, "没有帧记录"):
            next(read_rows(example_cache()[:1]))


class TrackingTests(unittest.TestCase):
    source = VideoSource("test-video", 25, 640, 480)
    bike = [100, 100, 160, 180, 0.9, 0]
    other_bike = [200, 100, 260, 180, 0.9, 0]

    def frame(self, index, rows, source=None):
        return FrameDetections(source or self.source, index, index / 25, rows)

    def test_result_adapter_and_cache_have_identical_outputs(self):
        with EXAMPLE.open(encoding="utf-8") as stream:
            cached = list(read_cache(stream))

        def direct_frames():
            for frame in cached:
                result = SimpleNamespace(
                    orig_shape=(frame.source.height, frame.source.width),
                    boxes=SimpleNamespace(data=torch.from_numpy(frame.detections)),
                )
                yield detections_from_result(result, frame.source, frame.frame_id, frame.timestamp_s)

        replay, direct, exported = io.StringIO(), io.StringIO(), io.StringIO()
        write_tracks(iter(cached), Tracker(), replay)
        write_tracks(direct_frames(), Tracker(), direct, exported)
        self.assertEqual(replay.getvalue(), direct.getvalue())
        self.assertEqual(list(map(json.loads, exported.getvalue().splitlines())), example_cache())

    def test_replay_cli_avoids_loading_detector_and_preserves_existing_output(self):
        output = EXAMPLE.with_name(f".tracks-{uuid4().hex}.jsonl")
        self.addCleanup(output.unlink, missing_ok=True)
        with patch("tracker.video.load_detector") as detector:
            arguments = ["--cache", str(EXAMPLE), "--output", str(output)]
            self.assertEqual(main(arguments), 0)
            original = output.read_bytes()
            records = [json.loads(line) for line in original.decode("utf-8").splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(records[-1]["tracks"][0]["track_id"], 1)
            detector.assert_not_called()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(arguments)
            self.assertEqual(output.read_bytes(), original)

    def test_confirmation_loss_recovery_expiration_and_source_reset(self):
        tracker = Tracker(TrackingConfig(min_hits=2, max_age=2), debug_predictions=True)
        self.assertEqual(tracker.update(self.frame(0, [self.bike])).tracks, ())
        self.assertEqual(tracker.update(self.frame(1, [self.bike])).tracks[0].track_id, 1)
        lost = tracker.update(self.frame(2, []))
        self.assertEqual(lost.tracks, ())
        self.assertEqual(lost.predictions[0].provenance, "prediction")
        self.assertEqual(tracker.update(self.frame(3, [self.bike])).tracks[0].track_id, 1)
        tracker.update(self.frame(4, []))
        tracker.update(self.frame(5, []))
        self.assertEqual(tracker.update(self.frame(6, [self.bike])).tracks, ())
        self.assertEqual(tracker.update(self.frame(7, [self.bike])).tracks[0].track_id, 2)
        different = VideoSource("next-video", 25, 640, 480)
        self.assertEqual(tracker.update(self.frame(0, [self.bike], different)).tracks, ())
        self.assertEqual(tracker.update(self.frame(1, [self.bike], different)).tracks[0].track_id, 1)

    def test_invalid_frame_does_not_advance_or_expire_tracks(self):
        tracker = Tracker(TrackingConfig(min_hits=1, max_age=1))
        tracker.update(self.frame(0, [self.bike]))
        with self.assertRaises(ValueError):
            tracker.update(self.frame(2, []))
        self.assertEqual(tracker.update(self.frame(1, [self.bike])).tracks[0].track_id, 1)

    def test_tentative_track_is_removed_after_first_miss(self):
        tracker = Tracker(TrackingConfig(min_hits=2))
        tracker.update(self.frame(0, [self.bike]))
        self.assertEqual(tracker.update(self.frame(1, [])).stats.live_tracks, 0)
        self.assertEqual(tracker.update(self.frame(2, [self.bike])).tracks, ())
        self.assertEqual(tracker.update(self.frame(3, [self.bike])).tracks[0].track_id, 2)

    def test_batch_correction_matches_scalar_kalman_updates(self):
        kalman = TrackingKalman()
        original = np.array([[200, 150, 0.5, 50], [30, 500, 1.2, 120], [220, 200, 0.7, 20]])
        means, covariances = map(np.stack, zip(*(kalman.initiate(row) for row in original)))
        for shift in (0.0, 1.0, 5.0, -2.0, 10.0):
            means, covariances = kalman.multi_predict(means, covariances)
            measurements = original + shift * np.array([1.0, -0.25, 0.001, 0.1])
            scalar_means, scalar_covariances = map(np.stack, zip(*(
                kalman.update(mean, cov, measurement)
                for mean, cov, measurement in zip(means, covariances, measurements)
            )))
            means, covariances = kalman.update_batch(means, covariances, measurements)
            np.testing.assert_allclose(means, scalar_means, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(covariances, scalar_covariances, rtol=1e-10, atol=1e-10)

    def test_assignment_uses_only_valid_edges_and_does_not_lose_feasible_pairs(self):
        ious = np.array([[0.99, 0.31], [0.30, 0.29]])
        rows, cols = _assign_iou(ious, ious >= 0.3)
        self.assertEqual(set(zip(rows, cols)), {(0, 1), (1, 0)})
        rows, cols = _assign_iou(ious, ious > 1)
        self.assertEqual((len(rows), len(cols)), (0, 0))
        rows, cols = _assign_iou(ious[:, :1], ious[:, :1] >= 0.3)
        self.assertEqual(set(zip(rows, cols)), {(0, 0)})

    def test_two_tracks_cannot_consume_one_detection_and_class_is_gated(self):
        tracker = Tracker(TrackingConfig(min_hits=1))
        tracker.update(self.frame(0, [self.bike, self.bike]))
        result = tracker.update(self.frame(1, [self.bike, [*self.bike[:5], 1]]))
        self.assertEqual([(t.track_id, t.detection_index) for t in result.tracks], [(1, 0), (3, 1)])

    def test_capacity_and_original_detection_indices(self):
        config = TrackingConfig(min_hits=1, max_tracks=1, max_detections=2)
        third_bike = [300, 100, 360, 180, 0.9, 0]
        rows = [[*self.bike[:4], 0.1, 0], self.other_bike, self.bike, third_bike]
        tracker = Tracker(config)
        result = tracker.update(self.frame(0, rows))
        self.assertEqual(result.tracks[0].detection_index, 1)
        self.assertEqual(result.stats.truncated_candidates, 1)
        self.assertEqual(result.stats.rejected_new_tracks, 1)
        for index in range(1, 101):
            result = tracker.update(self.frame(index, rows))
            self.assertEqual(result.stats.live_tracks, 1)
            self.assertLessEqual(result.stats.association_rows, 1)
            self.assertLessEqual(result.stats.association_columns, 2)
            self.assertEqual(result.tracks[0].track_id, 1)
            self.assertEqual(result.tracks[0].detection_index, 1)

    def mock_video(self, timestamps):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [(True, image)] * len(timestamps) + [(False, None)]
        metadata = {
            cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FRAME_COUNT: len(timestamps),
        }
        pts = iter(timestamps)
        capture.get.side_effect = lambda key: next(pts) if key == cv2.CAP_PROP_POS_MSEC else metadata[key]
        result = SimpleNamespace(orig_shape=(480, 640), boxes=SimpleNamespace(data=torch.tensor([self.bike])))
        return capture, Mock(return_value=result)

    def test_video_entry_exports_a_replayable_cache_and_profiles_paced_processing(self):
        output, cache, report, replay = [
            EXAMPLE.with_name(f".{uuid4().hex}-{kind}.jsonl") for kind in ("tracks", "cache", "report", "replay")
        ]
        for path in (output, cache, report, replay):
            self.addCleanup(path.unlink, missing_ok=True)
        capture, detector = self.mock_video([120.0, 160.0, 200.0])
        with patch("cv2.VideoCapture", return_value=capture), patch("tracker.video.load_detector", return_value=detector):
            self.assertEqual(main([
                "--video", str(EXAMPLE), "--weights", "unused.pt", "--output", str(output),
                "--save-cache", str(cache), "--report", str(report), "--profile-frames", "2",
            ]), 0)
        capture.release.assert_called_once()
        self.assertEqual(detector.call_count, 3)
        self.assertEqual(main(["--cache", str(cache), "--output", str(replay)]), 0)
        self.assertEqual(output.read_bytes(), replay.read_bytes())
        stats = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(stats["mode"], "paced_video")
        self.assertEqual(stats["frames_processed"], 3)
        self.assertEqual(stats["latency_samples"], 2)
        self.assertGreaterEqual(stats["end_to_end_latency_ms"]["p95"], stats["tracking_latency_ms"]["p95"])
        self.assertEqual(stats["dropped_frames"], 0)

    def test_video_rejects_variable_timebase(self):
        capture, detector = self.mock_video([0.0, 50.0])
        with patch("cv2.VideoCapture", return_value=capture):
            frames = video_detections(EXAMPLE, detector)
            self.assertEqual(next(frames).frame_id, 0)
            with self.assertRaisesRegex(ValueError, "固定帧率"):
                next(frames)
        capture.release.assert_called_once()

    def test_synthetic_report_scopes_measurements_to_tracking(self):
        result = benchmark(TrackingConfig(max_tracks=2, max_detections=3), frames=4, warmup=3, repeat=1, fps=30)
        self.assertEqual(result["scope"], "tracking_core_only")
        for case in result["cases"].values():
            report = case["runs"][0]
            self.assertEqual(report["latency_samples"], 4)
            self.assertIsNone(report["end_to_end_latency_ms"])
            self.assertEqual(report["measured_frames"], 4)
            self.assertGreater(report["peak_sampled_rss_bytes"], 0)
        pressure = result["cases"]["over_capacity"]["runs"][0]
        self.assertGreater(pressure["workload_totals"]["truncated_candidates"], 0)
        self.assertGreater(pressure["workload_totals"]["rejected_new_tracks"], 0)
