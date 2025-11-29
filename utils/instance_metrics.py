"""
Unified Instance Segmentation Evaluation Metrics

Provides consistent evaluation metrics for all instance segmentation models:
- Mask2Former
- Mask R-CNN

Metrics:
- IoU (Intersection over Union)
- Precision, Recall, F1 Score
- AP@50, AP@75, mAP (COCO-style)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary masks
    
    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)
    
    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks
    
    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)
    
    Returns:
        Dice score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0.0
    return float(2 * intersection / total)


def extract_instance_masks(instance_map: np.ndarray, background_ids: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    Extract individual binary masks from instance map
    
    Args:
        instance_map: (H, W) array with instance IDs
        background_ids: List of IDs to treat as background (default: [0, -1, 65535])
    
    Returns:
        List of binary masks
    """
    # Convert to int32 if float (e.g., from Mask2Former output)
    if instance_map.dtype in [np.float32, np.float64]:
        instance_map = instance_map.astype(np.int32)
    
    if background_ids is None:
        # Common background values: 0, -1 (which becomes 65535 in uint16)
        background_ids = [0, -1, 65535, 255]
    
    instance_ids = np.unique(instance_map)
    
    # Filter out background IDs
    instance_ids = [i for i in instance_ids if i not in background_ids]
    
    masks = []
    for inst_id in instance_ids:
        mask = (instance_map == inst_id).astype(np.uint8)
        masks.append(mask)
    
    return masks


def match_instances(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match predicted instances to ground truth instances using greedy matching
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
        iou_threshold: Minimum IoU for a match
    
    Returns:
        matched_pairs: List of (pred_idx, gt_idx, iou)
        unmatched_pred: List of unmatched prediction indices
        unmatched_gt: List of unmatched ground truth indices
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return [], list(range(len(pred_masks))), list(range(len(gt_masks)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    matched_pairs = []
    used_preds = set()
    used_gts = set()
    
    # Greedy matching by highest IoU
    while True:
        if len(used_preds) == len(pred_masks) or len(used_gts) == len(gt_masks):
            break
        
        # Find best remaining match
        max_iou = 0
        best_pair = None
        
        for i in range(len(pred_masks)):
            if i in used_preds:
                continue
            for j in range(len(gt_masks)):
                if j in used_gts:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    best_pair = (i, j)
        
        if best_pair is None or max_iou < iou_threshold:
            break
        
        matched_pairs.append((best_pair[0], best_pair[1], max_iou))
        used_preds.add(best_pair[0])
        used_gts.add(best_pair[1])
    
    unmatched_pred = [i for i in range(len(pred_masks)) if i not in used_preds]
    unmatched_gt = [j for j in range(len(gt_masks)) if j not in used_gts]
    
    return matched_pairs, unmatched_pred, unmatched_gt


def compute_ap(
    pred_masks: List[np.ndarray],
    pred_scores: np.ndarray,
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision at given IoU threshold
    
    Args:
        pred_masks: List of predicted binary masks
        pred_scores: Confidence scores for each prediction
        gt_masks: List of ground truth binary masks
        iou_threshold: IoU threshold for positive match
    
    Returns:
        AP score
    """
    if len(gt_masks) == 0:
        return 1.0 if len(pred_masks) == 0 else 0.0
    if len(pred_masks) == 0:
        return 0.0
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_masks = [pred_masks[i] for i in sorted_indices]
    
    tp = np.zeros(len(pred_masks))
    fp = np.zeros(len(pred_masks))
    used_gt = set()
    
    for i, pred in enumerate(pred_masks):
        best_iou = 0
        best_gt = -1
        
        for j, gt in enumerate(gt_masks):
            if j in used_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        
        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            used_gt.add(best_gt)
        else:
            fp[i] = 1
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    recall = tp_cumsum / len(gt_masks)
    
    # Compute AP using 11-point interpolation (PASCAL VOC style)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        if len(p) > 0:
            ap += np.max(p) / 11
    
    return float(ap)


def compute_coco_ap(
    pred_masks: List[np.ndarray],
    pred_scores: np.ndarray,
    gt_masks: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute COCO-style AP metrics
    
    Args:
        pred_masks: List of predicted binary masks
        pred_scores: Confidence scores for each prediction
        gt_masks: List of ground truth binary masks
    
    Returns:
        Dictionary with AP@50, AP@75, mAP (AP@50:95)
    """
    # AP at different IoU thresholds
    ap50 = compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold=0.5)
    ap75 = compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold=0.75)
    
    # mAP: average over IoU thresholds 0.5:0.05:0.95 (COCO style)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for thresh in iou_thresholds:
        ap = compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold=thresh)
        aps.append(ap)
    
    mAP = float(np.mean(aps))
    
    return {
        'AP@50': ap50,
        'AP@75': ap75,
        'mAP': mAP
    }


def compute_instance_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_scores: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive instance segmentation metrics
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
        pred_scores: Optional confidence scores (for AP calculation)
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Handle edge cases
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'mean_iou': 1.0,
            'AP@50': 1.0,
            'AP@75': 1.0,
            'mAP': 1.0,
            'num_pred': 0,
            'num_gt': 0,
            'num_matched': 0
        }
    
    if len(gt_masks) == 0:
        return {
            'precision': 0.0,
            'recall': 1.0,  # No GT to miss
            'f1': 0.0,
            'mean_iou': 0.0,
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0,
            'num_pred': len(pred_masks),
            'num_gt': 0,
            'num_matched': 0
        }
    
    if len(pred_masks) == 0:
        return {
            'precision': 1.0,  # No false positives
            'recall': 0.0,
            'f1': 0.0,
            'mean_iou': 0.0,
            'AP@50': 0.0,
            'AP@75': 0.0,
            'mAP': 0.0,
            'num_pred': 0,
            'num_gt': len(gt_masks),
            'num_matched': 0
        }
    
    # Match instances
    matched, unmatched_pred, unmatched_gt = match_instances(
        pred_masks, gt_masks, iou_threshold
    )
    
    # Basic metrics
    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Mean IoU of matched instances
    mean_iou = np.mean([m[2] for m in matched]) if matched else 0.0
    
    metrics.update({
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_iou': float(mean_iou),
        'num_pred': len(pred_masks),
        'num_gt': len(gt_masks),
        'num_matched': tp
    })
    
    # AP metrics (require scores)
    if pred_scores is not None and len(pred_scores) == len(pred_masks):
        ap_metrics = compute_coco_ap(pred_masks, pred_scores, gt_masks)
        metrics.update(ap_metrics)
    else:
        # If no scores or size mismatch, use uniform scores (order-based)
        if len(pred_masks) > 0:
            uniform_scores = np.linspace(1, 0.5, len(pred_masks))
        else:
            uniform_scores = np.array([])
        ap_metrics = compute_coco_ap(pred_masks, uniform_scores, gt_masks)
        metrics.update(ap_metrics)
    
    return metrics


def compute_metrics_from_instance_maps(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    pred_scores: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics from instance maps (convenience function)
    
    Args:
        pred_map: (H, W) predicted instance map
        gt_map: (H, W) ground truth instance map
        pred_scores: Optional scores for each instance
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with all metrics
    """
    pred_masks = extract_instance_masks(pred_map)
    gt_masks = extract_instance_masks(gt_map)
    
    return compute_instance_metrics(pred_masks, gt_masks, pred_scores, iou_threshold)


class MetricsAggregator:
    """
    Aggregate metrics across multiple images
    """
    
    def __init__(self):
        self.all_metrics = []
    
    def add(self, metrics: Dict[str, float]):
        """Add metrics for one image"""
        self.all_metrics.append(metrics)
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary statistics"""
        if not self.all_metrics:
            return {}
        
        keys = self.all_metrics[0].keys()
        summary = {}
        
        for key in keys:
            values = [m[key] for m in self.all_metrics if key in m]
            if values:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
        
        # Total counts
        summary['total_pred'] = sum(m.get('num_pred', 0) for m in self.all_metrics)
        summary['total_gt'] = sum(m.get('num_gt', 0) for m in self.all_metrics)
        summary['total_matched'] = sum(m.get('num_matched', 0) for m in self.all_metrics)
        summary['num_images'] = len(self.all_metrics)
        
        return summary
    
    def print_summary(self):
        """Print formatted summary"""
        summary = self.compute_summary()
        
        print("\n" + "=" * 60)
        print("Instance Segmentation Evaluation Results")
        print("=" * 60)
        print(f"Images evaluated: {summary.get('num_images', 0)}")
        print(f"Total predictions: {summary.get('total_pred', 0)}")
        print(f"Total ground truth: {summary.get('total_gt', 0)}")
        print(f"Total matched: {summary.get('total_matched', 0)}")
        print("-" * 60)
        print(f"Precision:  {summary.get('precision_mean', 0):.4f} ± {summary.get('precision_std', 0):.4f}")
        print(f"Recall:     {summary.get('recall_mean', 0):.4f} ± {summary.get('recall_std', 0):.4f}")
        print(f"F1 Score:   {summary.get('f1_mean', 0):.4f} ± {summary.get('f1_std', 0):.4f}")
        print(f"Mean IoU:   {summary.get('mean_iou_mean', 0):.4f} ± {summary.get('mean_iou_std', 0):.4f}")
        print("-" * 60)
        print(f"AP@50:      {summary.get('AP@50_mean', 0):.4f} ± {summary.get('AP@50_std', 0):.4f}")
        print(f"AP@75:      {summary.get('AP@75_mean', 0):.4f} ± {summary.get('AP@75_std', 0):.4f}")
        print(f"mAP:        {summary.get('mAP_mean', 0):.4f} ± {summary.get('mAP_std', 0):.4f}")
        print("=" * 60)
        
        return summary


if __name__ == '__main__':
    # Test the metrics
    print("Testing instance segmentation metrics...")
    
    # Create dummy data
    pred_map = np.zeros((100, 100), dtype=np.int32)
    pred_map[10:40, 10:40] = 1
    pred_map[50:80, 50:80] = 2
    
    gt_map = np.zeros((100, 100), dtype=np.int32)
    gt_map[15:45, 15:45] = 1
    gt_map[55:85, 55:85] = 2
    
    pred_scores = np.array([0.9, 0.8])
    
    metrics = compute_metrics_from_instance_maps(pred_map, gt_map, pred_scores)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ Test completed!")
