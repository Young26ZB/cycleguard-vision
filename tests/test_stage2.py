"""第二阶段数据与计分的非训练检查。"""

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from PIL import Image

from audit_ebike import publish_dataset, scan_dataset, write_json
from ebike_diagnostics import REVIEW_REASONS, load_manifest, score_diagnostics, sha256_file


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()).resolve() / f"ebike-test-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)


class ManifestTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.name = "images/test/example.png"
        image = self.root / self.name
        image.parent.mkdir(parents=True)
        Image.new("RGB", (120, 100)).save(image)
        self.path = self.root / "manifest.json"
        self.record = dict(
            width=120, height=100, sha256=sha256_file(image), training_overlap=True,
            lighting="day", status="ready", targets=[], ignore=[], review=[],
        )

    def load(self, name=None):
        manifest = dict(schema_version=1, dataset_version="fixture", images={name or self.name: self.record})
        self.path.write_text(json.dumps(manifest), encoding="utf-8")
        return load_manifest(self.path, self.root)

    def test_verified_empty_and_pending_reasons(self):
        self.assertEqual(self.load()["images"][self.name]["targets"], [])
        self.record.update(
            status="pending",
            review=[dict(reason=r, location="路口", evidence="轮廓不完整", conclusion=None) for r in REVIEW_REASONS],
        )
        self.assertEqual(len(self.load()["images"][self.name]["review"]), 3)
        self.record["status"] = "ready"
        with self.assertRaisesRegex(ValueError, "状态与未决"):
            self.load()

    def test_reject_invalid_records(self):
        with self.assertRaisesRegex(ValueError, "相对图片路径"):
            self.load("../example.png")
        for key, value, error in [
            ("width", 121, "尺寸"),
            ("sha256", "0" * 64, "SHA-256"),
            ("targets", None, "列表"),
            ("ignore", [[0, 0, 121, 2]], "超出图像"),
            ("ignore", [[1, 0, 1, 2]], "面积为零"),
            ("ignore", [[0, 0, 1, float("nan")]], "有限数值"),
        ]:
            with self.subTest(key=key, value=value):
                original = self.record[key]
                self.record[key] = value
                with self.assertRaisesRegex(ValueError, error):
                    self.load()
                self.record[key] = original

    def test_scoring_precedence_thresholds_and_coverage(self):
        self.record.update(
            targets=[
                dict(box=[0, 0, 40, 40], occlusion="none"),
                dict(box=[80, 60, 120, 100], occlusion="partial"),
            ],
            ignore=[[14, 0, 54, 40]],
        )
        manifest = self.load()
        manifest["images"]["pending"] = {
            "status": "pending", "lighting": "night", "training_overlap": False, "review": [{"reason": "uncertain_class"}],
        }
        result = score_diagnostics(manifest, {
            self.name: [
                dict(box=[0, 0, 40, 20], confidence=0.9),  # IoU 恰为 0.50
                dict(box=[10, 0, 50, 40], confidence=0.8),  # 同时覆盖目标和忽略区
                dict(box=[34, 0, 59, 40], confidence=0.7),  # IoA 恰为 0.80
                dict(box=[0, 60, 10, 70], confidence=0.25),  # 置信度恰好达标
                dict(box=[80, 60, 120, 100], confidence=0.24),
            ],
            "pending": [dict(box=[0, 0, 0, 0], confidence=1)],  # 整图跳过
        })
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "duplicates", "ignored")], [1, 2, 1, 1, 1])
        self.assertEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["duplicate_rate"], 1 / 3)
        self.assertEqual(result["localization_error_median"], 0.5)
        self.assertEqual(result["localization_pair_count"], 1)
        self.assertEqual(result["pending_images"], 1)
        self.assertEqual([p["outcome"] for p in result["frames"][self.name]["predictions"]],
                         ["tp", "duplicate", "ignored", "fp"])
        self.assertEqual(result["frames"][self.name]["targets"][1]["relative_area"], 1600 / 12000)

    def test_empty_and_missing_predictions(self):
        manifest = self.load()
        with self.assertRaisesRegex(ValueError, "不对应"):
            score_diagnostics(manifest, {})
        empty = score_diagnostics(manifest, {self.name: []})
        for key in ("precision", "recall", "duplicate_rate", "localization_error_median"):
            self.assertIsNone(empty[key])

    def test_evaluation_entrypoint_with_fixed_predictions(self):
        from evaluate_model import main

        self.record["targets"] = [dict(box=[0, 0, 10, 10], occlusion="none")]
        self.load()
        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixed weights identifier")
        model = Mock()
        model.names = {0: "ebike", 1: "other"}
        model.model = SimpleNamespace(end2end=True)
        boxes = SimpleNamespace(
            xyxy=Mock(**{"tolist.return_value": [[0, 0, 10, 10], [50, 50, 60, 60]]}),
            conf=Mock(**{"tolist.return_value": [0.25, 0.9]}),
            cls=Mock(**{"tolist.return_value": [0, 1]}),
        )
        model.predict.return_value = [SimpleNamespace(orig_shape=(100, 120), boxes=boxes)]
        args = ["evaluate_model.py", "--weights", str(weights), "--project", str(self.root), "--name", "diag"]
        diagnostic = args + ["--diagnostic-manifest", str(self.path), "--data-root", str(self.root)]
        with patch("sys.argv", diagnostic), patch("evaluate_model.YOLO", return_value=model), redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        report = json.loads((self.root / "diag/diagnostic-metrics.json").read_text(encoding="utf-8"))
        self.assertEqual((report["tp"], report["fp"], report["scored_images"]), (1, 0, 1))
        self.assertTrue(report["training_overlap"])
        self.assertEqual(report["inference"]["postprocess"], {"mode": "end2end", "nms_iou": None})
        self.assertLess(model.predict.call_args.kwargs["conf"], report["thresholds"]["confidence"])
        metrics = dict(precision=0.5, recall=0.5, f1=0.5, map50=0.5, map50_95=0.25)
        with (
            patch("sys.argv", args + ["--iou", "0.7", "--split", "test"]),
            patch("evaluate_model.evaluate_model", return_value=metrics) as legacy,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(main(), 0)
        old_args = legacy.call_args.args[0]
        self.assertEqual((old_args.iou, old_args.split, old_args.conf), (0.7, "test", 0.001))


class AuditTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.source = self.root / "source"
        self.output = self.root / "derived"
        self.review = self.root / "review.json"
        for split in ("train", "val"):
            (self.source / "images" / split).mkdir(parents=True)
            (self.source / "labels" / split).mkdir(parents=True)
        for name in ("train/a.png", "train/negative.png", "train/negative.jpg", "train/empty.png", "val/v.png"):
            Image.new("RGB", (10, 10)).save(self.source / "images" / name)
        self.valid = "0 0.5 0.5 0.2 0.2"
        (self.source / "labels/train/a.txt").write_text(self.valid + "\n0 0.5 0.5 0 0.2\n", encoding="utf-8")
        (self.source / "labels/train/classes.txt").write_text("ebike\n", encoding="utf-8")
        (self.source / "labels/train/empty.txt").touch()
        (self.source / "labels/val/v.txt").touch()

    def test_read_only_audit(self):
        _, report = scan_dataset(self.source)
        self.assertEqual(report["splits"]["train"], dict(
            images=4, zero_width=1, missing_label=2, shared_label_path=2, empty_labels=1,
        ))
        self.assertEqual(report["issues"]["images/train/a.png"]["issues"][0]["line"], 2)
        self.assertEqual(report["issues"]["images/train/negative.jpg"]["status"], "pending")
        self.assertEqual(report["orphan_labels"], [])
        self.assertFalse((self.source / "labels/train/negative.txt").exists())

    def test_derived_publish_requires_confirmed_decisions_and_preserves_source(self):
        _, report = scan_dataset(self.source)
        source_file = self.source / "labels/train/a.txt"
        before = sha256_file(source_file)
        write_json(self.review, {"schema_version": 1, "reviews": {}})
        with self.assertRaisesRegex(ValueError, "不完整"):
            publish_dataset(self.source, self.output, self.review)
        self.assertFalse(self.output.exists())
        self.review.unlink()

        approvals = {}
        for name, entry in report["issues"].items():
            labels = [self.valid] if name.endswith("/a.png") else []
            approvals[name] = dict(
                status="annotated" if labels else "background", labels=labels, reviewed_by="fixture",
                evidence="人工复核示例", image_sha256=entry["image_sha256"], label_sha256=entry["label_sha256"],
            )
        write_json(self.review, {"schema_version": 1, "reviews": approvals})
        config = publish_dataset(self.source, self.output, self.review)
        self.assertIn("path: .", config.read_text(encoding="utf-8"))
        mapping = json.loads((self.output / "source-map.json").read_text(encoding="utf-8"))["images"]
        self.assertEqual(len(mapping), 5)
        one, two = (mapping["images/train/negative." + suffix]["label"] for suffix in ("png", "jpg"))
        self.assertNotEqual(one, two)
        self.assertEqual((self.output / one).read_bytes(), b"")
        self.assertEqual((self.output / two).read_bytes(), b"")
        self.assertEqual(sha256_file(source_file), before)
        self.assertFalse((self.source / "labels/train/negative.txt").exists())
