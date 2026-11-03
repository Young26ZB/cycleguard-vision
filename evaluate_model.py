#!/usr/bin/env python3
"""
Model Evaluation Script for NMS-Free YOLOv11 Electric Bicycle Detection.

This script generates comprehensive evaluation metrics including:
- Confusion Matrix
- Precision, Recall, F1-Score
- mAP@0.5, mAP@0.5:0.95
- Precision-Recall Curve
- F1-Confidence Curve
- Per-class metrics

Usage:
    python evaluate_model.py --weights best.pt --data ebike.yaml
    python evaluate_model.py --weights best.pt --data ebike.yaml --save-plots

Author: NMS-FreeYOLO Project
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from train_ebike import get_data_config, resolve_data_config
from ultralytics import YOLO

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not found. Plotting will be disabled.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NMS-Free YOLOv11 Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data settings
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to trained model weights (.pt file)")
    parser.add_argument("--data", "-d", type=str, default="ebike.yaml", help="Dataset configuration file path")

    # Evaluation settings
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--data-root", type=str, default="", help="Override dataset root on this machine")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for metrics")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="", help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")

    # Output settings
    parser.add_argument("--project", type=str, default="runs/evaluate", help="Output directory")
    parser.add_argument("--name", type=str, default="ebike_eval", help="Experiment name")
    parser.add_argument("--save-plots", action="store_true", help="Save evaluation plots")
    parser.add_argument("--save-json", action="store_true", default=True, help="Save metrics as JSON")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    return parser.parse_args()


class MetricsCalculator:
    """Calculate and store evaluation metrics."""

    def __init__(self, num_classes: int = 1, class_names: list | None = None):
        """Initialize metrics calculator.

        Args:
            num_classes: Number of classes.
            class_names: List of class names.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        self.all_predictions = []
        self.all_targets = []
        self.all_confidences = []
        self.all_ious = []

    def update_confusion_matrix(
        self,
        pred_classes: np.ndarray,
        true_classes: np.ndarray,
        pred_conf: np.ndarray,
        iou_matrix: np.ndarray,
        iou_threshold: float = 0.5,
    ):
        """Update confusion matrix with batch predictions.

        Args:
            pred_classes: Predicted class indices.
            true_classes: Ground truth class indices.
            pred_conf: Prediction confidences.
            iou_matrix: IoU matrix between predictions and ground truths.
            iou_threshold: IoU threshold for matching.
        """
        # Match predictions to ground truths
        if len(pred_classes) == 0:
            # All ground truths are false negatives (background)
            for tc in true_classes:
                self.confusion_matrix[self.num_classes, int(tc)] += 1
            return

        if len(true_classes) == 0:
            # All predictions are false positives (background)
            for pc in pred_classes:
                self.confusion_matrix[int(pc), self.num_classes] += 1
            return

        # Find matches based on IoU
        matched_gt = set()
        matched_pred = set()

        # Sort predictions by confidence
        sorted_idx = np.argsort(-pred_conf)

        for pred_idx in sorted_idx:
            if len(true_classes) == 0:
                break

            # Find best matching ground truth
            ious = iou_matrix[pred_idx]
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                # True positive or class confusion
                pred_class = int(pred_classes[pred_idx])
                true_class = int(true_classes[best_gt_idx])
                self.confusion_matrix[pred_class, true_class] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
            else:
                # False positive (background)
                pred_class = int(pred_classes[pred_idx])
                self.confusion_matrix[pred_class, self.num_classes] += 1

        # Unmatched ground truths are false negatives
        for gt_idx, tc in enumerate(true_classes):
            if gt_idx not in matched_gt:
                self.confusion_matrix[self.num_classes, int(tc)] += 1

    def compute_metrics(self) -> dict:
        """Compute all metrics from confusion matrix.

        Returns:
            Dictionary containing all metrics.
        """
        metrics = {}

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.class_names):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

        metrics["per_class"] = per_class

        # Overall metrics (macro average)
        all_precision = [v["precision"] for v in per_class.values()]
        all_recall = [v["recall"] for v in per_class.values()]
        all_f1 = [v["f1"] for v in per_class.values()]

        metrics["precision"] = float(np.mean(all_precision))
        metrics["recall"] = float(np.mean(all_recall))
        metrics["f1"] = float(np.mean(all_f1))

        # Total counts
        total_tp = sum(v["tp"] for v in per_class.values())
        total_fp = sum(v["fp"] for v in per_class.values())
        total_fn = sum(v["fn"] for v in per_class.values())

        metrics["total_tp"] = total_tp
        metrics["total_fp"] = total_fp
        metrics["total_fn"] = total_fn

        # Micro average
        micro_precision = total_tp / (total_tp + total_fp + 1e-10)
        micro_recall = total_tp / (total_tp + total_fn + 1e-10)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10)

        metrics["micro_precision"] = float(micro_precision)
        metrics["micro_recall"] = float(micro_recall)
        metrics["micro_f1"] = float(micro_f1)

        return metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: list, save_path: str | None = None, normalize: bool = True
):
    """Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array.
        class_names: List of class names.
        save_path: Path to save the plot.
        normalize: Whether to normalize the matrix.
    """
    if not HAS_PLOTTING:
        print("Plotting not available. Install matplotlib and seaborn.")
        return

    # Add background class
    labels = [*class_names, "Background"]

    # Normalize if requested
    if normalize:
        cm = confusion_matrix.astype("float") / (confusion_matrix.sum(axis=0, keepdims=True) + 1e-10)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm = confusion_matrix
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, save_path: str | None = None):
    """Plot Precision-Recall curve.

    Args:
        precision: Precision values.
        recall: Recall values.
        ap: Average precision value.
        save_path: Path to save the plot.
    """
    if not HAS_PLOTTING:
        return

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, "b-", linewidth=2, label=f"AP={ap:.3f}")
    plt.fill_between(recall, precision, alpha=0.3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"PR curve saved to: {save_path}")
    plt.close()


def plot_f1_confidence_curve(
    f1_scores: np.ndarray, confidences: np.ndarray, best_conf: float, best_f1: float, save_path: str | None = None
):
    """Plot F1-Confidence curve.

    Args:
        f1_scores: F1 scores at different confidence thresholds.
        confidences: Confidence thresholds.
        best_conf: Best confidence threshold.
        best_f1: Best F1 score.
        save_path: Path to save the plot.
    """
    if not HAS_PLOTTING:
        return

    plt.figure(figsize=(10, 8))
    plt.plot(confidences, f1_scores, "b-", linewidth=2)
    plt.axvline(x=best_conf, color="r", linestyle="--", label=f"Best Conf={best_conf:.3f}, F1={best_f1:.3f}")
    plt.xlabel("Confidence")
    plt.ylabel("F1 Score")
    plt.title("F1-Confidence Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"F1-Confidence curve saved to: {save_path}")
    plt.close()


def plot_metrics_summary(metrics: dict, save_path: str | None = None):
    """Plot metrics summary bar chart.

    Args:
        metrics: Dictionary of metrics.
        save_path: Path to save the plot.
    """
    if not HAS_PLOTTING:
        return

    # Extract key metrics
    metric_names = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    metric_values = [
        metrics.get("precision", 0),
        metrics.get("recall", 0),
        metrics.get("f1", 0),
        metrics.get("map50", 0),
        metrics.get("map50_95", 0),
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f39c12"])

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Model Performance Summary")
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metrics summary saved to: {save_path}")
    plt.close()


def evaluate_model(args):
    """Run model evaluation.

    Args:
        args: Parsed command line arguments.

    Returns:
        Dictionary containing all evaluation results.
    """
    print("=" * 60)
    print("NMS-Free YOLOv11 Model Evaluation")
    print("=" * 60)

    # Check weights file
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Create output directory
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data = resolve_data_config(get_data_config(args.data), args.data_root, args.project)

    print(f"\nModel weights: {weights_path}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print(f"Output directory: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(str(weights_path))

    # Get class names
    class_names = list(model.names.values()) if hasattr(model, "names") else ["ebike"]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Run validation
    print("\nRunning validation...")
    print("-" * 60)

    val_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "split": args.split,
        "verbose": args.verbose,
        "save_json": True,
        "plots": args.save_plots,
    }

    if args.device:
        val_args["device"] = args.device

    results = model.val(**val_args)

    # Extract metrics from results
    metrics = {}

    # Basic metrics
    metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0))
    metrics["recall"] = float(results.results_dict.get("metrics/recall(B)", 0))
    metrics["map50"] = float(results.results_dict.get("metrics/mAP50(B)", 0))
    metrics["map50_95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0))

    # Calculate F1 score
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = float(2 * p * r / (p + r + 1e-10))

    # Fitness score (weighted combination)
    metrics["fitness"] = float(0.1 * metrics["map50"] + 0.9 * metrics["map50_95"])

    # Additional info
    metrics["num_classes"] = num_classes
    metrics["class_names"] = class_names
    metrics["model_path"] = str(weights_path)
    metrics["data_config"] = args.data
    metrics["image_size"] = args.imgsz
    metrics["conf_threshold"] = args.conf
    metrics["iou_threshold"] = args.iou
    metrics["split"] = args.split
    metrics["timestamp"] = datetime.now().isoformat()

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'Precision':<20} {metrics['precision']:.4f}")
    print(f"{'Recall':<20} {metrics['recall']:.4f}")
    print(f"{'F1 Score':<20} {metrics['f1']:.4f}")
    print(f"{'mAP@0.5':<20} {metrics['map50']:.4f}")
    print(f"{'mAP@0.5:0.95':<20} {metrics['map50_95']:.4f}")
    print(f"{'Fitness':<20} {metrics['fitness']:.4f}")

    # Per-class results if multiple classes
    if num_classes > 1:
        print("\nPer-class Results:")
        print("-" * 50)
        for i, name in enumerate(class_names):
            class_ap = results.box.ap50[i] if hasattr(results.box, "ap50") else 0
            print(f"  {name}: AP@0.5 = {class_ap:.4f}")

    # Save plots if requested
    if args.save_plots and HAS_PLOTTING:
        print("\nGenerating plots...")

        # Metrics summary
        plot_metrics_summary(metrics, str(output_dir / "metrics_summary.png"))

        # Copy YOLO-generated plots if they exist
        yolo_plots_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
        if yolo_plots_dir and yolo_plots_dir.exists():
            print(f"YOLO plots available at: {yolo_plots_dir}")

    # Save metrics as JSON
    if args.save_json:
        json_path = output_dir / "metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to: {json_path}")

    # Generate text report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("NMS-Free YOLOv11 Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n")
        f.write(f"Model: {metrics['model_path']}\n")
        f.write(f"Dataset: {metrics['data_config']}\n")
        f.write(f"Split: {metrics['split']}\n")
        f.write(f"Image Size: {metrics['image_size']}\n")
        f.write(f"Confidence Threshold: {metrics['conf_threshold']}\n")
        f.write(f"IoU Threshold: {metrics['iou_threshold']}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Performance Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision:     {metrics['precision']:.4f}\n")
        f.write(f"Recall:        {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:      {metrics['f1']:.4f}\n")
        f.write(f"mAP@0.5:       {metrics['map50']:.4f}\n")
        f.write(f"mAP@0.5:0.95:  {metrics['map50_95']:.4f}\n")
        f.write(f"Fitness:       {metrics['fitness']:.4f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Classes\n")
        f.write("-" * 40 + "\n")
        for name in class_names:
            f.write(f"  - {name}\n")

    print(f"Report saved to: {report_path}")

    return metrics


def main():
    """Main entry point."""
    args = parse_args()

    try:
        metrics = evaluate_model(args)

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

        # Quick summary
        print("\n📊 Quick Summary:")
        print(f"   Precision: {metrics['precision']:.2%}")
        print(f"   Recall:    {metrics['recall']:.2%}")
        print(f"   F1 Score:  {metrics['f1']:.2%}")
        print(f"   mAP@0.5:   {metrics['map50']:.2%}")
        print(f"   mAP@0.5:0.95: {metrics['map50_95']:.2%}")

        return 0

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
