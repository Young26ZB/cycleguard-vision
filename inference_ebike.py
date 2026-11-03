#!/usr/bin/env python3
"""
NMS-Free YOLOv11 Inference Script for Electric Bicycle Detection.

This script performs inference using a trained NMS-Free YOLOv11 model
for detecting electric bicycles in images, videos, or camera streams.

Usage:
    python inference_ebike.py --weights runs/train/ebike_nms_free/weights/best.pt --source image.jpg
    python inference_ebike.py --weights best.pt --source video.mp4
    python inference_ebike.py --weights best.pt --source 0  # webcam

Author: NMS-FreeYOLO Project
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NMS-Free YOLOv11 Inference for Electric Bicycle Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to trained model weights (.pt file)")

    # Input source
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="Input source: image path, video path, directory, URL, or camera index",
    )

    # Inference settings
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (if not end2end)")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--device", type=str, default="", help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision")

    # Output settings
    parser.add_argument("--save", action="store_true", default=True, help="Save results")
    parser.add_argument("--save-txt", action="store_true", help="Save results as text")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence in text labels")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detections")
    parser.add_argument("--project", type=str, default="runs/detect", help="Save directory")
    parser.add_argument("--name", type=str, default="ebike_inference", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing results")

    # Visualization settings
    parser.add_argument("--show", action="store_true", help="Show results on screen")
    parser.add_argument("--show-labels", action="store_true", default=True, help="Show labels")
    parser.add_argument("--show-conf", action="store_true", default=True, help="Show confidence")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")

    # Stream settings
    parser.add_argument("--stream", action="store_true", help="Treat input as stream")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame stride")

    # Advanced settings
    parser.add_argument("--classes", nargs="+", type=int, help="Filter by class")
    parser.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="Augmented inference")
    parser.add_argument("--visualize", action="store_true", help="Visualize features")
    parser.add_argument("--retina-masks", action="store_true", help="High-res masks")

    return parser.parse_args()


def run_inference(args):
    """Run inference with the specified arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Inference results.
    """
    print("=" * 60)
    print("NMS-Free YOLOv11 Inference for Electric Bicycle Detection")
    print("=" * 60)

    # Check weights file
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"\nModel weights: {weights_path}")
    print(f"Input source: {args.source}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(str(weights_path))

    # Check if model is end2end (NMS-free)
    is_end2end = getattr(model.model, "end2end", False)
    print(f"End-to-end (NMS-free) mode: {is_end2end}")

    # Prepare inference arguments
    predict_args = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "half": args.half,
        "save": args.save,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "show": args.show,
        "show_labels": args.show_labels,
        "show_conf": args.show_conf,
        "line_width": args.line_width,
        "stream": args.stream,
        "vid_stride": args.vid_stride,
        "classes": args.classes,
        "agnostic_nms": args.agnostic_nms,
        "augment": args.augment,
        "visualize": args.visualize,
        "retina_masks": args.retina_masks,
    }

    # Add device if specified
    if args.device:
        predict_args["device"] = args.device

    # Run inference
    print("\nRunning inference...")
    print("-" * 60)

    start_time = time.time()
    results = model.predict(**predict_args)
    end_time = time.time()

    # Process results
    total_detections = 0
    for r in results:
        total_detections += len(r.boxes)

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"\nTotal detections: {total_detections}")
    print(f"Inference time: {end_time - start_time:.2f}s")

    if args.save:
        save_dir = Path(args.project) / args.name
        print(f"Results saved to: {save_dir}")

    return results


def run_benchmark(args):
    """Run inference benchmark to measure speed.

    Args:
        args: Parsed command line arguments.
    """
    print("\nRunning speed benchmark...")

    # Load model
    model = YOLO(args.weights)

    # Warmup
    print("Warming up...")
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    if args.device:
        dummy_input = dummy_input.to(args.device)
    for _ in range(10):
        model.predict(dummy_input, verbose=False)

    # Benchmark
    print("Benchmarking...")
    num_runs = 100
    times = []

    for _ in range(num_runs):
        start = time.time()
        model.predict(dummy_input, verbose=False)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    print("\nBenchmark Results:")
    print(f"  Average inference time: {avg_time * 1000:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Image size: {args.imgsz}x{args.imgsz}")


def main():
    """Main entry point."""
    args = parse_args()

    # Run inference
    results = run_inference(args)

    # Print summary for each result
    print("\nDetection Summary:")
    for i, r in enumerate(results):
        print(f"\n  Image {i + 1}:")
        print(f"    Detections: {len(r.boxes)}")
        if len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy()
            print(f"    Confidence range: {confs.min():.3f} - {confs.max():.3f}")
            print(f"    Average confidence: {confs.mean():.3f}")


if __name__ == "__main__":
    main()
