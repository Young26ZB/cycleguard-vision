# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""NMS-Free Detection Loss with Dual Label Assignment and Repulsion Loss."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .loss import BboxLoss, DFLoss
from .repulsion_loss import RepulsionLossV2
from .tal import bbox2dist


class NMSFreeBboxLoss(nn.Module):
    """Extended Bounding Box Loss with Repulsion Loss for NMS-Free detection.
    
    Combines standard IoU/DFL loss with Repulsion Loss to explicitly
    separate overlapping predictions in crowded scenes.
    
    Attributes:
        dfl_loss (DFLoss): Distribution Focal Loss for box regression.
        repulsion_loss (RepulsionLossV2): Repulsion loss for box separation.
        use_repulsion (bool): Whether to use repulsion loss.
        repulsion_weight (float): Weight for repulsion loss.
    """
    
    def __init__(self, reg_max: int = 16, use_repulsion: bool = True, repulsion_weight: float = 0.5):
        """Initialize NMSFreeBboxLoss.
        
        Args:
            reg_max (int): DFL maximum regularization value.
            use_repulsion (bool): Enable repulsion loss.
            repulsion_weight (float): Weight for repulsion loss component.
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.use_repulsion = use_repulsion
        self.repulsion_weight = repulsion_weight
        if use_repulsion:
            self.repulsion_loss = RepulsionLossV2(alpha=0.5, beta=1.0)
    
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute IoU, DFL, and Repulsion losses.
        
        Args:
            pred_dist: Predicted distance distribution.
            pred_bboxes: Predicted bounding boxes.
            anchor_points: Anchor point coordinates.
            target_bboxes: Target bounding boxes.
            target_scores: Target scores.
            target_scores_sum: Sum of target scores.
            fg_mask: Foreground mask.
            
        Returns:
            Tuple of (iou_loss, dfl_loss, repulsion_loss).
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # Repulsion loss
        if self.use_repulsion and self.repulsion_weight > 0 and fg_mask.any():
            loss_rep = self.repulsion_loss(
                pred_bboxes.float() * stride_tensor.float(),
                gt_bboxes,
                fg_mask,
                target_gt_idx,
                mask_gt,
            )
            loss_rep = loss_rep * self.repulsion_weight
        else:
            loss_rep = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, loss_rep


class NMSFreeDetectionLoss:
    """NMS-Free Detection Loss with Dual Label Assignment strategy.
    
    This loss implements the dual label assignment strategy:
    - One-to-Many (O2M): Multiple predictions can match one GT (for training efficiency)
    - One-to-One (O2O): Each GT matches only one prediction (for NMS-free inference)
    
    The combined loss ensures the model learns efficiently while producing
    unique predictions that don't require NMS post-processing.
    
    Attributes:
        o2m_assigner (TaskAlignedAssigner): One-to-many label assigner.
        o2o_assigner (TaskAlignedAssigner): One-to-one label assigner.
        bbox_loss (NMSFreeBboxLoss): Box regression loss.
        bce (nn.BCEWithLogitsLoss): Binary cross entropy for classification.
        use_repulsion (bool): Whether to use repulsion loss.
    
    Examples:
        >>> model = YOLO('yolo11n.yaml')
        >>> loss_fn = NMSFreeDetectionLoss(model)
        >>> preds = model(images)
        >>> loss, loss_items = loss_fn(preds, batch)
    """
    
    def __init__(
        self,
        model,
        o2m_topk: int = 10,
        o2o_topk: int = 1,
        use_repulsion: bool = True,
        repulsion_weight: float = 0.5,
        o2m_weight: float = 1.0,
        o2o_weight: float = 1.0,
    ):
        """Initialize NMSFreeDetectionLoss.
        
        Args:
            model: YOLO model (must be de-paralleled).
            o2m_topk (int): Top-k for one-to-many assignment.
            o2o_topk (int): Top-k for one-to-one assignment (usually 1).
            use_repulsion (bool): Enable repulsion loss.
            repulsion_weight (float): Weight for repulsion loss.
            o2m_weight (float): Weight for one-to-many branch loss.
            o2o_weight (float): Weight for one-to-one branch loss.
        """
        device = next(model.parameters()).device
        h = SimpleNamespace(**model.args) if isinstance(model.args, dict) else model.args
        
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        
        self.use_dfl = m.reg_max > 1
        self.use_repulsion = use_repulsion
        self.o2m_weight = o2m_weight
        self.o2o_weight = o2o_weight
        
        # One-to-Many assigner (traditional, multiple preds per GT)
        self.o2m_assigner = TaskAlignedAssigner(
            topk=o2m_topk, 
            num_classes=self.nc, 
            alpha=0.5, 
            beta=6.0
        )
        
        # One-to-One assigner (NMS-free, single pred per GT)
        self.o2o_assigner = TaskAlignedAssigner(
            topk=o2o_topk, 
            num_classes=self.nc, 
            alpha=0.5, 
            beta=6.0
        )
        
        # Bounding box loss with repulsion
        self.bbox_loss = NMSFreeBboxLoss(
            m.reg_max, 
            use_repulsion=use_repulsion,
            repulsion_weight=repulsion_weight
        ).to(device)
        
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    
    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for loss computation."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted bounding boxes."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def compute_loss_branch(
        self,
        feats: list[torch.Tensor],
        batch: dict[str, torch.Tensor],
        assigner: TaskAlignedAssigner,
        branch_name: str = "o2m",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for a single branch (O2M or O2O).
        
        Args:
            feats: Feature maps from detection head.
            batch: Batch data containing targets.
            assigner: Label assigner to use.
            branch_name: Name of branch for logging.
            
        Returns:
            Tuple of (total_loss, loss_items).
        """
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, rep
        
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # Predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # Label assignment
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        # Box loss
        if fg_mask.sum():
            loss[0], loss[2], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor,
                gt_bboxes,
                target_gt_idx,
                mask_gt,
            )
        
        # Apply hyperparameter weights
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        # Repulsion loss weight is already applied in bbox_loss
        
        return loss.sum() * batch_size, loss.detach()
    
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total NMS-Free detection loss.
        
        For end2end models, this computes both O2M and O2O losses.
        For standard models, it computes O2M loss with repulsion.
        
        Args:
            preds: Model predictions.
            batch: Batch data.
            
        Returns:
            Tuple of (total_loss, loss_items).
        """
        preds = preds[1] if isinstance(preds, tuple) else preds
        
        # Check if this is an end2end model with dual outputs
        if isinstance(preds, dict) and "one2many" in preds and "one2one" in preds:
            # Dual branch mode
            o2m_feats = preds["one2many"]
            o2o_feats = preds["one2one"]
            
            # Compute losses for both branches
            o2m_loss, o2m_items = self.compute_loss_branch(o2m_feats, batch, self.o2m_assigner, "o2m")
            o2o_loss, o2o_items = self.compute_loss_branch(o2o_feats, batch, self.o2o_assigner, "o2o")
            
            # Combine losses
            total_loss = self.o2m_weight * o2m_loss + self.o2o_weight * o2o_loss
            loss_items = self.o2m_weight * o2m_items + self.o2o_weight * o2o_items
            
            return total_loss, loss_items
        else:
            # Single branch mode (standard training)
            feats = preds if isinstance(preds, list) else [preds]
            loss, loss_items = self.compute_loss_branch(feats, batch, self.o2m_assigner, "o2m")
            return loss, loss_items


class E2EDetectLossWithRepulsion:
    """End-to-End Detection Loss with Repulsion for crowded scenes.
    
    This is a drop-in replacement for E2EDetectLoss that adds
    repulsion loss to handle high-density detection scenarios.
    
    Attributes:
        one2many (NMSFreeDetectionLoss): One-to-many branch loss.
        one2one (NMSFreeDetectionLoss): One-to-one branch loss.
    """
    
    def __init__(self, model, use_repulsion: bool = True, repulsion_weight: float = 0.5):
        """Initialize E2EDetectLossWithRepulsion.
        
        Args:
            model: YOLO model.
            use_repulsion (bool): Enable repulsion loss.
            repulsion_weight (float): Weight for repulsion loss.
        """
        self.one2many = NMSFreeDetectionLoss(
            model, 
            o2m_topk=10, 
            o2o_topk=10,
            use_repulsion=use_repulsion,
            repulsion_weight=repulsion_weight,
        )
        self.one2one = NMSFreeDetectionLoss(
            model, 
            o2m_topk=1, 
            o2o_topk=1,
            use_repulsion=use_repulsion,
            repulsion_weight=repulsion_weight,
        )
    
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute combined loss for both branches."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many.compute_loss_branch(one2many, batch, self.one2many.o2m_assigner, "o2m")
        one2one = preds["one2one"]
        loss_one2one = self.one2one.compute_loss_branch(one2one, batch, self.one2one.o2o_assigner, "o2o")
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
