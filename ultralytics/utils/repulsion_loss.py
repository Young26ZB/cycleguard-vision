# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Repulsion Loss for handling crowded object detection scenarios."""

from __future__ import annotations

import torch
import torch.nn as nn


class RepulsionLoss(nn.Module):
    """Repulsion Loss for crowded scene detection.
    
    This loss function addresses the issue of detection box adhesion in 
    crowded scenes by explicitly penalizing:
    1. RepGT Loss: Overlap between predicted boxes and non-matched ground truth
    2. RepBox Loss: Overlap between predicted boxes themselves
    
    Particularly effective for high-density electric bicycle detection where
    vehicles are closely packed during rush hours.
    
    Reference:
        Repulsion Loss: Detecting Pedestrians in a Crowd (CVPR 2018)
    
    Attributes:
        sigma_repgt (float): Smoothing parameter for RepGT loss.
        sigma_repbox (float): Smoothing parameter for RepBox loss.
        iou_thresh (float): IoU threshold to identify neighboring boxes.
        use_smooth (bool): Whether to use smooth L1 loss variant.
    
    Examples:
        >>> rep_loss = RepulsionLoss(sigma_repgt=0.9, sigma_repbox=0.0)
        >>> pred_boxes = torch.randn(10, 4)  # 10 predicted boxes
        >>> target_boxes = torch.randn(5, 4)  # 5 GT boxes
        >>> matched_idx = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 4, 3])
        >>> loss = rep_loss(pred_boxes, target_boxes, matched_idx)
    """
    
    def __init__(
        self,
        sigma_repgt: float = 0.9,
        sigma_repbox: float = 0.0,
        iou_thresh: float = 0.5,
        use_smooth: bool = True,
    ):
        """Initialize RepulsionLoss.
        
        Args:
            sigma_repgt (float): Sigma for RepGT IoG smoothing. Default: 0.9.
            sigma_repbox (float): Sigma for RepBox IoU smoothing. Default: 0.0.
            iou_thresh (float): IoU threshold for neighboring detection. Default: 0.5.
            use_smooth (bool): Use Smooth L1 variant. Default: True.
        """
        super().__init__()
        self.sigma_repgt = sigma_repgt
        self.sigma_repbox = sigma_repbox
        self.iou_thresh = iou_thresh
        self.use_smooth = use_smooth
    
    def smooth_ln(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Smooth L1-like loss for repulsion.
        
        When sigma = 0, this becomes -ln(1-x).
        When sigma > 0, this becomes a smooth variant.
        
        Args:
            x: Input tensor (IoU or IoG values).
            sigma: Smoothing parameter.
            
        Returns:
            Smoothed loss values.
        """
        x = x.clamp(0, 1 - torch.finfo(x.dtype).eps)
        if sigma == 0:
            # Standard repulsion: -ln(1 - IoU)
            return -torch.log(1 - x + 1e-10)
        else:
            # Smooth variant
            cond = x < sigma
            loss = torch.where(
                cond,
                -torch.log(1 - x + 1e-10),
                (x - sigma) / (1 - sigma) - x.new_tensor(1 - sigma).log()
            )
            return loss
    
    def compute_iog(
        self, 
        pred_boxes: torch.Tensor, 
        gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute Intersection over Ground-truth (IoG).
        
        IoG = Intersection / Area_of_GT
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format.
            gt_boxes: Ground truth boxes (M, 4) in xyxy format.
            
        Returns:
            IoG matrix of shape (N, M).
        """
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]
        
        if N == 0 or M == 0:
            return torch.zeros((N, M), device=pred_boxes.device)
        
        # Expand for broadcasting
        pred_boxes = pred_boxes.unsqueeze(1)  # (N, 1, 4)
        gt_boxes = gt_boxes.unsqueeze(0)  # (1, M, 4)
        
        # Compute intersection
        inter_x1 = torch.max(pred_boxes[..., 0], gt_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], gt_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], gt_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], gt_boxes[..., 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # GT area
        gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * \
                  (gt_boxes[..., 3] - gt_boxes[..., 1])
        
        # IoG
        iog = inter_area / (gt_area + 1e-10)
        
        return iog
    
    def compute_iou_matrix(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix for a set of boxes.
        
        Args:
            boxes: Boxes tensor of shape (N, 4) in xyxy format.
            
        Returns:
            IoU matrix of shape (N, N).
        """
        N = boxes.shape[0]
        if N == 0:
            return torch.zeros((0, 0), device=boxes.device)
        
        # Compute areas
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Expand for pairwise computation
        boxes1 = boxes.unsqueeze(1)  # (N, 1, 4)
        boxes2 = boxes.unsqueeze(0)  # (1, N, 4)
        
        # Intersection
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = area.unsqueeze(1)
        area2 = area.unsqueeze(0)
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-10)
        
        return iou
    
    def repgt_loss(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        matched_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RepGT (Repulsion Ground Truth) loss.
        
        Penalizes predicted boxes that overlap with non-matched ground truths,
        forcing separation between predictions and non-target objects.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format.
            gt_boxes: Ground truth boxes (M, 4) in xyxy format.
            matched_gt_idx: Index of matched GT for each prediction (N,).
            
        Returns:
            Scalar RepGT loss.
        """
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]
        
        if N == 0 or M <= 1:
            return pred_boxes.sum() * 0.0
        
        # Compute IoG between predictions and all GTs
        iog = self.compute_iog(pred_boxes, gt_boxes)  # (N, M)
        
        # Create mask to exclude matched GT for each prediction
        # matched_gt_idx[i] is the GT index that pred[i] is matched to
        mask = torch.ones_like(iog, dtype=torch.bool)
        for i in range(N):
            if matched_gt_idx[i] >= 0 and matched_gt_idx[i] < M:
                mask[i, matched_gt_idx[i]] = False
        
        # Get IoG with non-matched GTs only
        iog_non_matched = iog * mask
        
        # For each prediction, get max IoG with non-matched GTs
        max_iog, _ = iog_non_matched.max(dim=1)
        
        # Compute loss only for predictions with significant overlap
        valid_mask = max_iog > 0
        if not valid_mask.any():
            return pred_boxes.sum() * 0.0
        
        loss = self.smooth_ln(max_iog[valid_mask], self.sigma_repgt)
        
        return loss.mean()
    
    def repbox_loss(
        self,
        pred_boxes: torch.Tensor,
        matched_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RepBox (Repulsion Box) loss.
        
        Penalizes overlap between predicted boxes that are matched to 
        different ground truths, forcing separation between predictions.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format.
            matched_gt_idx: Index of matched GT for each prediction (N,).
            
        Returns:
            Scalar RepBox loss.
        """
        N = pred_boxes.shape[0]
        
        if N <= 1:
            return pred_boxes.sum() * 0.0
        
        # Compute pairwise IoU
        iou_matrix = self.compute_iou_matrix(pred_boxes)  # (N, N)
        
        # Create mask: only consider pairs matched to different GTs
        # and exclude self-comparison (diagonal)
        idx1 = matched_gt_idx.unsqueeze(1)  # (N, 1)
        idx2 = matched_gt_idx.unsqueeze(0)  # (1, N)
        diff_gt_mask = (idx1 != idx2)  # Different GT targets
        
        # Exclude diagonal (self-comparison)
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=pred_boxes.device)
        mask = diff_gt_mask & diag_mask
        
        # Get IoU for valid pairs
        iou_valid = iou_matrix * mask
        
        # Get max IoU for each prediction with others matched to different GTs
        max_iou, _ = iou_valid.max(dim=1)
        
        # Compute loss for predictions with significant overlap
        valid_mask = max_iou > 0
        if not valid_mask.any():
            return pred_boxes.sum() * 0.0
        
        loss = self.smooth_ln(max_iou[valid_mask], self.sigma_repbox)
        
        return loss.mean()
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        matched_gt_idx: torch.Tensor,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total repulsion loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format.
            gt_boxes: Ground truth boxes (M, 4) in xyxy format.
            matched_gt_idx: Index of matched GT for each prediction (N,).
            weights: Tuple of (repgt_weight, repbox_weight).
            
        Returns:
            Tuple of (total_loss, repgt_loss, repbox_loss).
        """
        repgt = self.repgt_loss(pred_boxes, gt_boxes, matched_gt_idx)
        repbox = self.repbox_loss(pred_boxes, matched_gt_idx)
        
        total = weights[0] * repgt + weights[1] * repbox
        
        return total, repgt, repbox


class RepulsionLossV2(nn.Module):
    """Batched matching-aware RepGT + RepBox in image coordinates.

    Real GT boxes (including unmatched objects) are used for RepGT. Predictions
    assigned to the same GT never repel one another. Reduced precision inputs
    are promoted to float32 for stable IoU/logarithm computation.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0, sigma: float = 0.5):
        super().__init__()
        if not 0 <= sigma < 1 or alpha < 0 or beta < 0:
            raise ValueError("Require 0 <= sigma < 1 and nonnegative repulsion weights")
        self.alpha, self.beta, self.sigma = alpha, beta, sigma
        self.loss = RepulsionLoss(sigma_repgt=0.9, sigma_repbox=sigma)

    def forward(self, pred_boxes, gt_boxes, fg_mask, matched_gt_idx, mask_gt):
        """Return batch-mean loss; boxes are xyxy in a shared image space.

        Shapes: predictions (B,N,4), GT (B,M,4), foreground and matched GT
        indices (B,N), valid GT mask (B,M,1). No image-to-image interactions.
        """
        pred_boxes, gt_boxes = pred_boxes.float(), gt_boxes.float()
        total = pred_boxes.sum() * 0.0
        for b in range(pred_boxes.shape[0]):
            fg = fg_mask[b].bool()
            valid = mask_gt[b].reshape(-1).bool()
            if not fg.any() or not valid.any():
                continue
            # Remap padded GT indices, so noncontiguous masks are also valid.
            remap = torch.full((len(valid),), -1, device=pred_boxes.device, dtype=torch.long)
            remap[valid] = torch.arange(int(valid.sum()), device=pred_boxes.device)
            matched = remap[matched_gt_idx[b][fg].long()]
            if (matched < 0).any():
                raise ValueError("Foreground prediction assigned to a padded GT")
            pred = pred_boxes[b][fg]
            gt = gt_boxes[b][valid]
            repgt = self.loss.repgt_loss(pred, gt, matched)
            repbox = self.loss.repbox_loss(pred, matched)
            total = total + self.alpha * repgt + self.beta * repbox
        return total / max(pred_boxes.shape[0], 1)
