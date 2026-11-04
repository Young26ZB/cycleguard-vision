"""Stage-one regression checks using synthetic tensors only; no optimizer or training.

Run: python -m unittest tests.test_stage1 -v
Set STAGE1_DEVICE=cuda on the training server to also check CUDA autocast.
"""
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from train_ebike import ROOT, get_model_config, get_data_config, resolve_data_config
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.block import BNContrastiveHead, ContrastiveHead
from ultralytics.nn.modules.dysample import DySample
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.nms_free_loss import NMSFreeBboxLoss
from ultralytics.utils.repulsion_loss import RepulsionLossV2

torch.set_num_threads(2)


class StageOneTests(unittest.TestCase):
    def test_configuration(self):
        for scale in 'nsm':
            path = get_model_config('yolo11-ebike.yaml', scale)
            from ultralytics.nn.tasks import yaml_model_load
            self.assertEqual(yaml_model_load(path)['scale'], scale)
        with self.assertRaises(FileNotFoundError):
            get_model_config('yolo11-typo.yaml', 'n')
        cfg = get_cfg(overrides={'use_repulsion': False, 'repulsion_weight': 0.25})
        self.assertFalse(cfg.use_repulsion)
        self.assertEqual(cfg.repulsion_weight, 0.25)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            config = Path(get_data_config('ebike.yaml'))
            source = Path(tmp) / config.relative_to(ROOT)
            source.parent.mkdir(parents=True)
            source.write_bytes(config.read_bytes())
            data_root = Path(tmp) / 'ebike'
            (data_root / 'images/train').mkdir(parents=True)
            (data_root / 'images/val').mkdir()
            resolved = resolve_data_config(str(source), '', tmp)
            self.assertEqual(Path(YAML.load(resolved)['path']), data_root)
            self.assertEqual(source.read_bytes(), config.read_bytes())

    def test_world_head_preserves_contrastive_batch_norm_option(self):
        YOLO(get_model_config('yolo11-ebike.yaml', 'n'), verbose=False)
        for variant, expected_head in [('world', ContrastiveHead), ('worldv2', BNContrastiveHead)]:
            with self.subTest(variant=variant):
                model = YOLO(f'yolov8s-{variant}.yaml', verbose=False)
                head = model.model.model[-1]
                self.assertIsInstance(head.cv4[0], expected_head)
                self.assertFalse(head.end2end)

    def test_zero_offset_sampling(self):
        for style in ('lp', 'pl'):
            for h, w in ((1, 1), (3, 5)):
                module = DySample(8, style=style, use_dyscope=True)
                for p in module.offset.parameters():
                    torch.nn.init.zeros_(p)
                x = torch.randn(2, 8, h, w, requires_grad=True)
                result = module(x)
                expected = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_repulsion_matching_and_numerics(self):
        loss = RepulsionLossV2()
        pred = torch.tensor([[[0., 0., 2., 2.], [0., 0., 2., 2.]]], requires_grad=True)
        gt = torch.tensor([[[0., 0., 2., 2.], [1., 0., 3., 2.]]])
        fg = torch.ones(1, 2, dtype=torch.bool)
        valid = torch.ones(1, 2, 1, dtype=torch.bool)
        same = torch.zeros(1, 2, dtype=torch.long)
        # One GT, two O2M positives: neither RepGT nor RepBox may repel them.
        self.assertEqual(loss(pred, gt[:, :1], fg, same, valid[:, :1]).item(), 0)
        # A single prediction must still repel a nonmatching GT.
        self.assertGreater(loss(pred[:, :1], gt, fg[:, :1], same[:, :1], valid).item(), 0)
        different = torch.tensor([[0, 1]])
        value = loss(pred, gt, fg, different, valid)
        self.assertTrue(torch.isfinite(value))
        torch.testing.assert_close(value, loss(pred * 4 + 10, gt * 4 + 10, fg, different, valid))
        value.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        empty = loss(pred, gt, ~fg, different, valid)
        self.assertEqual(empty.item(), 0)
        empty.backward()
        torch.testing.assert_close(value, loss(pred.repeat(2, 1, 1), gt.repeat(2, 1, 1),
                                               fg.repeat(2, 1), different.repeat(2, 1), valid.repeat(2, 1, 1)))
        # Empty image / no GT, including a padded zero box.
        self.assertEqual(loss(pred, gt[:, :0], ~fg, different, valid[:, :0]).item(), 0)

    def test_repulsion_uses_image_coordinates(self):
        module = NMSFreeBboxLoss(reg_max=1)
        captured = []
        module.repulsion_loss.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
        pred = torch.tensor([[[0., 0., 2., 2.], [0., 0., 2., 2.]]], requires_grad=True)
        strides = torch.tensor([[8.], [16.]])
        result = module(torch.zeros(1, 2, 4), pred, torch.zeros(2, 2), pred.detach(),
                        torch.ones(1, 2, 1), torch.tensor(2.), torch.ones(1, 2, dtype=torch.bool),
                        strides, pred.detach() * strides, torch.tensor([[0, 1]]),
                        torch.ones(1, 2, 1, dtype=torch.bool))
        torch.testing.assert_close(captured[0], pred.detach() * strides)
        sum(result).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_pretrained_does_not_replace_architecture(self):
        from train_ebike import parse_args, train
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            data_root = Path(tmp) / 'dataset'
            (data_root / 'images/train').mkdir(parents=True)
            (data_root / 'images/val').mkdir()
            source = YOLO(get_model_config('yolo11.yaml', 'n'), verbose=False)
            weights = Path(tmp) / 'source.pt'
            source.save(weights)
            argv = ['train_ebike.py', '--model', 'yolo11-ebike.yaml', '--pretrained', str(weights),
                    '--data-root', str(data_root), '--project', tmp, '--check-only']
            with patch('sys.argv', argv), patch.object(YOLO, 'train') as training:
                self.assertIsNone(train(parse_args()))
                training.assert_not_called()

    def test_models_forward_backward_and_reload(self):
        device = torch.device(os.getenv('STAGE1_DEVICE', 'cpu'))
        for config in ('yolo11-ebike.yaml', 'yolo11-nms-free.yaml'):
            with self.subTest(config=config):
                yolo = YOLO(get_model_config(config, 'n'), verbose=False)
                model = yolo.model.to(device)
                # Creating another head must not mutate an existing model's mode.
                YOLO(get_model_config('yolo11.yaml', 'n'), verbose=False)
                self.assertTrue(model.model[-1].end2end)
                x = torch.rand(2, 3, 64, 64, device=device)
                for enabled in (False, True):
                    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, 'use_repulsion': enabled})
                    model.criterion = model.init_criterion()
                    for count in (0, 1, 2):
                        model.train()
                        model.zero_grad(set_to_none=True)
                        batch = {'img': x, 'batch_idx': torch.zeros(count, device=device),
                                 'cls': torch.zeros(count, 1, device=device),
                                 'bboxes': torch.tensor([[.5, .5, .6, .6], [.6, .5, .5, .5]],
                                                        device=device)[:count]}
                        total, items = model.loss(batch)
                        self.assertEqual(tuple(items.shape), (4,))
                        self.assertTrue(torch.isfinite(total).all())
                        if not enabled:
                            self.assertEqual(items[3].item(), 0)
                        total.sum().backward()
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                            if p.grad is not None))
                # Autocast is a synthetic gradient check, not a training step.
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                    total, _ = model.loss(batch)
                self.assertTrue(torch.isfinite(total).all())
                model.zero_grad(set_to_none=True)
                total.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval()
                with torch.no_grad():
                    expected, expected_raw = model(x)
                self.assertEqual(expected.shape[-1], 6)
                with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                    path = Path(tmp) / 'synthetic.pt'
                    yolo.save(path)
                    restored = YOLO(path, verbose=False).model.to(device).eval()
                    with torch.no_grad():
                        actual, actual_raw = restored(x)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(torch.isfinite(actual).all())
                    # FP16 checkpoint rounding can reorder tied TopK scores in
                    # an untrained head. Compare spatially ordered raw outputs.
                    for branch in ('one2many', 'one2one'):
                        for a, b in zip(actual_raw[branch], expected_raw[branch]):
                            torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)


if __name__ == '__main__':
    unittest.main()
