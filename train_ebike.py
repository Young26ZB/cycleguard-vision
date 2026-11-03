#!/usr/bin/env python3
"""
NMS-Free YOLOv11 Training Script for Electric Bicycle Detection.

This script trains a YOLOv11 model optimized for high-density electric bicycle
detection in urban rush hour scenarios. Features:
- End-to-end detection (NMS-Free)
- Dual Label Assignment (One-to-Many + One-to-One)
- DySample dynamic upsampling
- SimAM parameter-free attention
- Repulsion Loss for crowded scenes

Usage:
    python train_ebike.py --data path/to/ebike.yaml --epochs 300

Author: NMS-FreeYOLO Project
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NMS-Free YOLOv11 for Electric Bicycle Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolo11-ebike.yaml",
        help="Model configuration file or pretrained weights path",
    )
    parser.add_argument(
        "--model-scale",
        "-s",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model scale: n(ano), s(mall), m(edium), l(arge), x(large)",
    )

    # Dataset configuration
    parser.add_argument("--data", "-d", type=str, default="ebike.yaml", help="Dataset configuration file path")

    parser.add_argument("--data-root", type=str, default="", help="Override dataset root on this machine")
    parser.add_argument("--check-only", action="store_true", help="Build model and resolve configs without training")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="", help="Device to use (cuda:0, cpu, etc.)")

    # Optimizer settings
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")

    # Loss weights
    parser.add_argument("--box", type=float, default=7.5, help="Box loss weight")
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss weight")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss weight")
    parser.add_argument(
        "--use-repulsion", action="store_true", default=True, help="Enable Repulsion Loss for crowded scenes"
    )
    parser.add_argument("--no-repulsion", dest="use_repulsion", action="store_false", help="Disable Repulsion Loss")
    parser.add_argument("--repulsion-weight", type=float, default=0.5, help="Nonnegative Repulsion Loss weight")

    # Data augmentation
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-H augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-S augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-V augmentation")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear degrees")
    parser.add_argument("--flipud", type=float, default=0.0, help="Flip up-down probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.1, help="Mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.1, help="Copy-paste augmentation")

    # Output settings
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--name", type=str, default="ebike_nms_free", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment")
    parser.add_argument("--pretrained", type=str, default="", help="Pretrained weights path")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    # Validation settings
    parser.add_argument("--val", action="store_true", default=True, help="Run validation during training")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")

    # Advanced settings
    parser.add_argument("--cache", type=str, default="", help="Cache images (ram, disk, or empty)")
    parser.add_argument("--rect", action="store_true", help="Use rectangular training")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic last N epochs")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True, help="Use automatic mixed precision"
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction to train on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    return parser.parse_args()


def get_model_config(model_name: str, scale: str) -> str:
    """Get the appropriate model configuration file.

    Args:
        model_name: Model name or path.
        scale: Model scale (n, s, m, l, x).

    Returns:
        Path to model configuration file.
    """
    path = Path(model_name)
    if path.suffix == ".pt":
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return str(path.resolve())
    if not path.suffix:
        path = path.with_suffix(".yaml")
    # Resolve scale aliases only when their canonical YAML exists. Never fall
    # back to a different architecture after a misspelled model name.
    canonical_name = re.sub(r"^(yolo11)[nslmx](?=-|$)", r"\1", path.stem) + path.suffix
    candidates = [path, path.with_name(canonical_name)]
    if path.parent == Path("."):
        cfg_dir = ROOT / "ultralytics/cfg/models/11"
        candidates += [cfg_dir / path.name, cfg_dir / canonical_name]
    resolved = next((p.resolve() for p in candidates if p.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Model configuration does not exist: {model_name}")
    # yaml_model_load recognizes this alias, loads the canonical YAML and
    # stores the requested scale in model.yaml, including trainer rebuilds.
    stem = re.sub(r"^(yolo11)[nslmx](?=-|$)", r"\1", resolved.stem)
    if not re.match(r"^yolo11(?=-|$)", stem):
        raise ValueError("Use a yolo11[-variant].yaml filename for --model-scale")
    return str(resolved.with_name(re.sub(r"^yolo11", f"yolo11{scale}", stem) + resolved.suffix))


def get_data_config(data_name: str) -> str:
    """Get the appropriate dataset configuration file.

    Args:
        data_name: Dataset name or path.

    Returns:
        Path to dataset configuration file.
    """
    # Check if it's a path to existing file
    if Path(data_name).exists():
        return str(Path(data_name).resolve())

    # Check in ultralytics config directory
    config_dir = ROOT / "ultralytics" / "cfg" / "datasets"

    candidates = [
        config_dir / data_name,
        config_dir / f"{data_name}.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Dataset configuration does not exist: {data_name}")


def resolve_data_config(config: str, data_root: str, project: str) -> str:
    """Write a portable resolved copy, leaving the source dataset YAML intact.

    Relative roots are anchored at the source YAML, never global datasets_dir.
    """
    source = Path(config).resolve()
    data = YAML.load(source)
    root = Path(data_root).expanduser().resolve() if data_root else Path(data.get("path") or ".")
    if not root.is_absolute():
        root = (source.parent / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}; use --data-root")
    for split in ("train", "val"):
        entries = data.get(split)
        if not entries:
            raise ValueError(f"Dataset YAML must define {split}")
        for entry in entries if isinstance(entries, list) else [entries]:
            if not (root / entry).exists():
                raise FileNotFoundError(f"Missing {split} input: {root / entry}")
    data["path"] = str(root)
    token = hashlib.sha256(repr(data).encode()).hexdigest()[:12]
    target = Path(project).resolve() / "_configs" / f"{source.stem}-{token}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(target, data)
    return str(target)


def train(args):
    """Run training with the specified arguments.

    Args:
        args: Parsed command line arguments.
    """
    print("=" * 60)
    print("NMS-Free YOLOv11 Training for Electric Bicycle Detection")
    print("=" * 60)

    # Get model and data configuration paths
    model_config = get_model_config(args.model, args.model_scale)
    data_config = resolve_data_config(get_data_config(args.data), args.data_root, args.project)
    if args.repulsion_weight < 0:
        raise ValueError("--repulsion-weight must be nonnegative")
    if args.resume and (Path(model_config).suffix != ".pt" or args.pretrained):
        raise ValueError("--resume requires --model path/to/last.pt and no --pretrained")

    print(f"\nModel config: {model_config}")
    print(f"Data config: {data_config}")
    print(f"Model scale: {args.model_scale}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(
        f"Repulsion Loss: {'Enabled' if args.use_repulsion else 'Disabled'}"
        + (f" (weight={args.repulsion_weight})" if args.use_repulsion else "")
    )
    print()

    # Build the selected architecture first; pretrained weights must not replace it.
    model = YOLO(model_config)
    if args.pretrained:
        if not Path(args.pretrained).is_file():
            raise FileNotFoundError(f"Pretrained weights do not exist: {args.pretrained}")
        model.load(args.pretrained)
        source_head = model.ckpt["model"].model[-1]
        head = model.model.model[-1]
        if getattr(head, "end2end", False) and not hasattr(source_head, "one2one_cv2"):
            head.one2one_cv2.load_state_dict(head.cv2.state_dict())
            head.one2one_cv3.load_state_dict(head.cv3.state_dict())
            print("Initialized one-to-one head from transferred one-to-many weights")
    print(f"Effective scale: {model.model.yaml.get('scale', 'checkpoint')}")
    print(f"End-to-end: {model.model.end2end}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    if args.check_only:
        print("Configuration check complete; training was not started.")
        return None

    # Training configuration
    train_args = {
        "data": data_config,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "verbose": args.verbose,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "resume": args.resume,
        "amp": args.amp,
        "fraction": args.fraction,
        "cache": args.cache if args.cache else False,
        "rect": args.rect,
        "close_mosaic": args.close_mosaic,
        "val": args.val,
        "save_period": args.save_period,
        # Optimizer
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        # Loss weights
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "use_repulsion": args.use_repulsion,
        "repulsion_weight": args.repulsion_weight,
        # Augmentation
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
    }

    # Add device if specified
    if args.device:
        train_args["device"] = args.device

    # Start training
    print("\nStarting training...")
    print("-" * 60)

    results = model.train(**train_args)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")

    return results


def validate(model_path: str, data_config: str, args):
    """Run validation on a trained model.

    Args:
        model_path: Path to trained model weights.
        data_config: Path to dataset configuration.
        args: Parsed command line arguments.
    """
    print("\nRunning validation...")
    model = YOLO(model_path)

    results = model.val(
        data=resolve_data_config(get_data_config(data_config), args.data_root, args.project),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.25,
        iou=0.5,
        device=args.device if args.device else None,
    )

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Run training
    results = train(args)

    if results is None:
        return

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")


if __name__ == "__main__":
    main()
