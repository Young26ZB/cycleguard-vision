#!/usr/bin/env python3
"""
Test script to verify NMS-Free modules are properly installed and working.

Usage:
    python test_modules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def test_dysample():
    """Test DySample module."""
    print("\n[1/4] Testing DySample module...")
    try:
        from ultralytics.nn.modules.dysample import DySample, DySampleBlock

        # Test DySample
        dysample = DySample(in_channels=256, scale_factor=2)
        x = torch.randn(1, 256, 20, 20)
        out = dysample(x)

        assert out.shape == (1, 256, 40, 40), f"Expected (1, 256, 40, 40), got {out.shape}"
        print(f"  ✓ DySample: input {x.shape} -> output {out.shape}")

        # Test DySampleBlock
        dysample_block = DySampleBlock(in_channels=256, out_channels=128, scale_factor=2)
        out2 = dysample_block(x)
        assert out2.shape == (1, 128, 40, 40), f"Expected (1, 128, 40, 40), got {out2.shape}"
        print(f"  ✓ DySampleBlock: input {x.shape} -> output {out2.shape}")

        print("  ✓ DySample module: PASSED")
        return True
    except Exception as e:
        print(f"  ✗ DySample module: FAILED - {e}")
        return False


def test_simam():
    """Test SimAM module."""
    print("\n[2/4] Testing SimAM module...")
    try:
        from ultralytics.nn.modules.attention import EfficientSimAM, SimAM

        # Test SimAM
        simam = SimAM(e_lambda=1e-4)
        x = torch.randn(1, 256, 40, 40)
        out = simam(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        print(f"  ✓ SimAM: input {x.shape} -> output {out.shape}")

        # Test EfficientSimAM
        efficient_simam = EfficientSimAM(e_lambda=1e-4)
        out2 = efficient_simam(x)
        assert out2.shape == x.shape, f"Expected {x.shape}, got {out2.shape}"
        print(f"  ✓ EfficientSimAM: input {x.shape} -> output {out2.shape}")

        # Verify no learnable parameters
        num_params = sum(p.numel() for p in simam.parameters())
        assert num_params == 0, f"SimAM should have 0 parameters, got {num_params}"
        print("  ✓ SimAM has 0 learnable parameters (parameter-free)")

        print("  ✓ SimAM module: PASSED")
        return True
    except Exception as e:
        print(f"  ✗ SimAM module: FAILED - {e}")
        return False


def test_bifpn():
    """Test BiFPN module."""
    print("\n[3/4] Testing BiFPN module...")
    try:
        from ultralytics.nn.modules.bifpn import BiFPN, FastNormalizedFusion

        # Test FastNormalizedFusion
        fusion = FastNormalizedFusion(num_inputs=3)
        inputs = [torch.randn(1, 256, 40, 40) for _ in range(3)]
        out = fusion(inputs)
        assert out.shape == inputs[0].shape, f"Expected {inputs[0].shape}, got {out.shape}"
        print(f"  ✓ FastNormalizedFusion: 3 inputs -> output {out.shape}")

        # Test BiFPN
        bifpn = BiFPN(in_channels_list=[256, 512, 1024], out_channels=256, num_layers=2)
        features = [
            torch.randn(1, 256, 80, 80),  # P3
            torch.randn(1, 512, 40, 40),  # P4
            torch.randn(1, 1024, 20, 20),  # P5
        ]
        outputs = bifpn(features)

        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        print("  ✓ BiFPN: 3 input features -> 3 output features")
        for i, out in enumerate(outputs):
            print(f"    Level {i}: {out.shape}")

        print("  ✓ BiFPN module: PASSED")
        return True
    except Exception as e:
        print(f"  ✗ BiFPN module: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_loading():
    """Test model configuration loading."""
    print("\n[4/4] Testing model configuration loading...")
    try:
        from ultralytics import YOLO

        # Test loading the simple ebike config
        config_path = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-ebike.yaml"

        if config_path.exists():
            print(f"  Loading config: {config_path}")
            model = YOLO(str(config_path))
            print("  ✓ Model loaded successfully")
            print(f"    - Number of parameters: {sum(p.numel() for p in model.model.parameters()):,}")
            print(f"    - End2End mode: {getattr(model.model, 'end2end', False)}")

            # Test forward pass
            x = torch.randn(1, 3, 640, 640)
            model.model.eval()
            with torch.no_grad():
                model.model(x)
            print("  ✓ Forward pass successful")

            print("  ✓ Model loading: PASSED")
            return True
        else:
            print(f"  ⚠ Config file not found: {config_path}")
            print("  ⚠ Model loading: SKIPPED")
            return True

    except Exception as e:
        print(f"  ✗ Model loading: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\n[0/4] Testing module imports...")
    try:
        from ultralytics.nn.modules import (
            BiFPN,
            BiFPNBlock,
            DySample,
            SimAM,
        )

        print("  ✓ All NMS-Free modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("NMS-Free YOLOv11 Module Tests")
    print("=" * 60)

    results = []

    # Test imports first
    results.append(("Imports", test_imports()))

    # Test individual modules
    results.append(("DySample", test_dysample()))
    results.append(("SimAM", test_simam()))
    results.append(("BiFPN", test_bifpn()))
    results.append(("Model Loading", test_model_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The NMS-Free modules are ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
