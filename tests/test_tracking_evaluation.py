"""用已知跨帧 ID 核验离线消融和部署筛选协议。"""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from tracker.evaluate import evaluate, select_realtime_profile
from tracker.evaluation import GroundTruth, MOTSequence, RecoveryMetrics
from tracker.records import FrameDetections, FrameStats, FrameTracks, TrackObservation, VideoSource


def box(identity):
    return [identity * 40, 30, identity * 40 + 30, 80]


def annotations(index, visible, occluded=(), partial=()):
    return {
        "frame_id": index, "visible": [[identity, *box(identity), 0] for identity in visible],
        "occluded": list(occluded), "partial": list(partial),
    }


class RecoveryEvaluationTests(unittest.TestCase):
    source = VideoSource("recovery", 30, 320, 120)

    def test_separate_recovered_wrong_unmatched_and_detector_misses(self):
        metrics = RecoveryMetrics(0.5, (1, 3), 0.25)
        mot = MOTSequence()

        def observe(index, visible, occluded, detector_ids, track_ids, partial=()):
            frame = FrameDetections(self.source, index, index / 30, [[*box(i), 1, 0] for i in detector_ids])
            truth = GroundTruth.from_record(annotations(index, visible, occluded, partial), frame)
            result = FrameTracks(
                self.source.source_id, index, index / 30,
                tuple(TrackObservation(identity, 0, tuple(box(i)), j) for j, (i, identity) in enumerate(track_ids)),
                FrameStats(0, 0, 0, 0, 0, 0, 0),
            )
            mot.append(truth, result, metrics.observe(truth, frame, result))

        observe(0, [1, 2, 3, 4], [], [1, 2, 3, 4], [(1, 11), (2, 22), (3, 33), (4, 44)], partial=[1])
        observe(1, [], [1, 2, 3, 4], [], [])
        observe(2, [1, 2, 3, 4], [], [1, 2, 4], [(1, 11), (2, 44), (4, 55)])
        report = metrics.report()
        self.assertEqual(report["1-1"], report["all"])
        self.assertEqual(report["all"]["detector_misses"], 1)
        self.assertEqual(report["all"]["all_events"], {
            "events": 4, "recovered": 1, "wrong_reconnect": 2, "missing_anchor": 0, "unmatched": 1, "new_id": 0,
            "same_id_rate": 0.25, "wrong_reconnect_rate": 0.5,
        })
        self.assertEqual(report["all"]["with_detection"]["events"], 3)
        self.assertEqual(report["all"]["with_detection"]["unmatched"], 0)
        self.assertIsNone(report["4+"]["all_events"]["same_id_rate"])
        self.assertIsNone(report["4+"]["with_detection"]["wrong_reconnect_rate"])
        calibration = metrics.calibration_report()
        self.assertEqual(calibration["reappeared"]["missed"], 1)
        self.assertEqual(calibration["reappeared"]["score_bins"][-1]["count"], 3)
        self.assertEqual(calibration["partial"]["score_bins"][-1]["mean_iou_error"], 0)

        data = mot.data()
        self.assertEqual(data["num_gt_ids"], 4)
        self.assertEqual(data["num_tracker_ids"], 5)
        self.assertEqual(data["tracker_ids"][-1].tolist(), [0, 3, 4])
        self.assertEqual(data["similarity_scores"][1].shape, (0, 0))

    def test_unmarked_gap_and_new_sequence_do_not_create_events(self):
        metrics = RecoveryMetrics(0.5, (), 0.25)
        for index, visible, hidden in [(0, [1], []), (1, [], []), (2, [1], []), (3, [], [1])]:
            frame = FrameDetections(self.source, index, index / 30, [])
            truth = GroundTruth.from_record(annotations(index, visible, hidden), frame)
            metrics.observe(truth, frame, FrameTracks(
                self.source.source_id, index, index / 30, (), FrameStats(0, 0, 0, 0, 0, 0, 0),
            ))
        metrics.reset_sequence()
        frame = FrameDetections(self.source, 0, 0, [])
        truth = GroundTruth.from_record(annotations(0, [1]), frame)
        metrics.observe(truth, frame, FrameTracks(self.source.source_id, 0, 0, (), FrameStats(0, 0, 0, 0, 0, 0, 0)))
        self.assertEqual(metrics.report()["all"]["all_events"]["events"], 0)


class ProtocolTests(unittest.TestCase):
    protocol = {
        "iou": 0.5, "occlusion_bins": [1, 3], "tracking": {"min_hits": 1},
        "target_fps": 30, "tracking_p95_ms": 6,
        "quality_thresholds": {
            "min_idf1": 0.8, "min_assa": 0.8, "min_same_id_rate": 0.8,
            "max_wrong_reconnect_rate": 0.1, "max_id_switches": 1,
        },
    }
    weights = "a" * 64

    def setUp(self):
        self.directory = Path("tracker") / f".test-evaluation-{uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.directory.rmdir)

    def write(self, filename, *records):
        path = self.directory / filename
        if not path.exists():
            self.addCleanup(path.unlink, missing_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def test_validation_and_test_use_the_same_frozen_protocol(self):
        sequences = []
        for name in ("validation", "test"):
            source = VideoSource(name, 30, 320, 120)
            cache = self.write(
                name + "-cache.jsonl", source.cache_header(),
                *(FrameDetections(source, i, i / 30, [[*box(1), 0.95, 0]] if i != 1 else []).cache_record()
                  for i in range(3)),
            )
            truth = self.write(name + "-truth.jsonl", annotations(0, [1]), annotations(1, [], [1]), annotations(2, [1]))
            sequences.append({"id": name, "split": name, "cache": cache.name, "truth": truth.name})
        manifest = self.write("manifest.json", {"weights_sha256": self.weights, "sequences": sequences})
        protocol = self.write("protocol.json", self.protocol)
        validation = evaluate(manifest, protocol, "validation", events_only=True)
        frozen = self.write("validation.json", validation)
        test = evaluate(manifest, protocol, "test", events_only=True, validation_path=frozen)
        self.assertEqual(set(test["profiles"]), {
            "baseline", "nsa_only", "full", "without_ocm", "without_ocr", "without_oru", "without_nsa", "fixed_ocm",
        })
        self.assertEqual(set(test["inputs"]), {"test"})
        self.assertEqual(test["profiles"]["full"]["recovery"]["all"]["with_detection"]["recovered"], 1)
        self.assertIsNone(test["profiles"]["full"]["mot"])
        self.write(protocol.name, {**self.protocol, "tracking_p95_ms": 7})
        with self.assertRaisesRegex(ValueError, "不一致"):
            evaluate(manifest, protocol, "test", events_only=True, validation_path=frozen)

    def test_deployment_selects_lowest_latency_only_after_all_gates(self):
        source = asdict(VideoSource("test", 30, 320, 120))
        names = ("baseline", "full", "cached", "other_weights", "other_device", "low_quality")
        quality = {
            "mot": {"idf1": 0.9, "assa": 0.9, "id_switches": 0},
            "recovery": {"all": {"all_events": {"same_id_rate": 1.0, "wrong_reconnect_rate": 0.0}}},
        }
        report = {
            "registration": {"tracking": {}, "profiles": dict.fromkeys(names, {}), "weights_sha256": self.weights},
            "profiles": {**dict.fromkeys(names, quality), "low_quality": {**quality, "mot": None}},
            "inputs": {"test": {"source": source, "frames": 40}},
        }
        setup = {"environment": {"cpu": "synthetic"}, "device_argument": "cpu", "warmup_frames": 1, "sample_limit": 40}
        common = {
            **setup, "source": source, "frames_processed": 40, "mode": "paced_video", "weights_sha256": self.weights,
            "target_fps": 30, "dropped_frames": 0, "final_schedule_backlog_frames": 0,
            "observed_throughput_fps": 30.1, "tracker_config": {}, "tracking_latency_ms": {"p50": 3, "p95": 4, "p99": 5},
        }
        paths = {}
        for name, elapsed, extra in [
            ("baseline", 16, {}), ("full", 15, {}), ("cached", 5, {"mode": "tracking_core_only"}),
            ("other_weights", 6, {"weights_sha256": "b" * 64}), ("other_device", 7, {"device_argument": "0"}),
            ("low_quality", 8, {}),
        ]:
            run = {**common, **extra, "end_to_end_latency_ms": {"p50": elapsed - 1, "p95": elapsed, "p99": elapsed + 1}}
            paths[name] = {"test": self.write(f"{name}.json", run).name}
        timing = self.write("timing.json", {**setup, "profiles": paths})
        result = select_realtime_profile(report, self.protocol, timing)
        self.assertEqual(result["selected"], "full")
        for name in ("cached", "other_weights"):
            self.assertIn("test:incompatible_paced_video", result["profiles"][name]["failures"])
        self.assertIn("test:incompatible_measurement_setup", result["profiles"]["other_device"]["failures"])
        self.assertIn("quality:idf1", result["profiles"]["low_quality"]["failures"])


if __name__ == "__main__":
    unittest.main()
