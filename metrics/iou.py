import numpy as np
import torch 

def _to_numpy(x):
    """
    Convert input to numpy array.
    Args:
        x: Input tensor or array.
    Returns:
        Numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}.")

def _ensure_batch(x):
    """
    Ensure input has batch dimension.
    Args:
        x: Input array of shape (H, W) or (B, H, W).
    Returns:
        Array of shape (B, H, W).
    """
    if x.ndim == 2:
        x = x[None, ...]
    elif x.ndim != 3:
        raise ValueError(f"Expected input with 2 or 3 dimensions, got {x.ndim} dimensions.")
    return x


def binarize_heatmap(
    heatmaps,
    top_k_percent = 0.15,
    threshold = None,
):
    """
    Binarize heatmaps into binary masks.

    Args:
        heatmaps: Input heatmaps, shape (H, W) or (B, H, W).
        top_k_percent: Percentage of top pixels to consider, top_kth percentile is taken as threshold.
            Used if threshold is None.
        threshold: Fixed threshold to binarize heatmaps. If provided,
            top_k_percent is ignored.
    Returns:
        binarized: Binarized masks, shape (B, H, W).
    """
    heatmaps_np = _to_numpy(heatmaps)
    heatmaps_np = _ensure_batch(heatmaps_np)

    b, h, w = heatmaps_np.shape
    
    #Normalize heatmaps to [0, 1]
    heatmaps_flat = heatmaps_np.reshape(b, -1)
    map_mins = heatmaps_flat.min(axis=1, keepdims=True)
    map_maxs = heatmaps_flat.max(axis=1, keepdims=True)
    map_norm = (heatmaps_flat - map_mins) / np.maximum(map_maxs - map_mins, 1e-8)

    heatmaps_np = map_norm.reshape(b, h, w) # (B, H, W)

    binarized = np.zeros_like(heatmaps_np, dtype=np.uint8)

    for i in range(b):
        heatmap = heatmaps_np[i]
        if threshold is not None:
            binarized[i] = (heatmap >= threshold).astype(np.uint8)
        else:
            flat = heatmap.flatten()
            k = max(0, int((1 - top_k_percent) * flat.shape[0]))
            thresh_value = np.partition(flat, k)[k]
            binarized[i] = (heatmap >= thresh_value).astype(np.uint8)

    return binarized


def iou_binary(
    pred,
    target,
    eps = 1e-8,
):
    """
    Compute Intersection-over-Union (IoU) between binary masks.

    Args:
        pred: Predicted binary masks, shape (H, W) or (B, H, W).
        target: Ground truth binary masks, shape (H, W) or (B, H, W).
        eps: Small value to avoid division by zero.
    
    Returns:
        ious: Array of IoU scores for each sample, shape (B,).
    
    Notes:
        pred and target are expected to be binary (0 or 1).
    """
    pred_np = _to_numpy(pred).astype(np.float32)
    target_np = _to_numpy(target).astype(np.float32)

    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_np.shape}, target {target_np.shape}."
        )

    pred_np = _ensure_batch(pred_np)
    target_np = _ensure_batch(target_np)

    # binarize defensively in case inputs are not binary
    pred_bin = (pred_np > 0.5).astype(np.float32)
    target_bin = (target_np > 0.5).astype(np.float32)

    intersection = (pred_bin * target_bin).sum(axis=(1, 2))
    union = (pred_bin + target_bin - pred_bin * target_bin).sum(axis=(1, 2))

    ious = intersection / (union + eps)
    return ious

def ious_from_cams(
    cams,
    masks,
    top_k_percent = 0.15,
    eps = 1e-8,
):
    """
    Compute IoUs between CAMs and ground truth binary masks.

    Args:
        cams: Class Activation Maps, shape (B, H, W).
        masks: Ground truth binary masks, shape (B, H, W).
        top_k_percent: Percentage of top pixels in CAM to consider as predicted mask.
        eps: Small value to avoid division by zero.

    Returns:
        ious: Array of IoU scores for each sample, shape (B,).
    """
    if top_k_percent <= 0 or top_k_percent >= 1.0:
        raise ValueError("top_k_percent must be in the range (0, 1).")
    
    cams_np = _to_numpy(cams).astype(np.float32)
    masks_np = _to_numpy(masks).astype(np.float32)

    if cams_np.shape != masks_np.shape:
        raise ValueError(
            f"Shape mismatch: cams {cams_np.shape}, masks {masks_np.shape}."
        )

    # binarize CAM into predicted masks
    pred_masks = binarize_heatmap(
        cams_np,
        top_k_percent=top_k_percent
    )

    gt_masks = _ensure_batch(masks_np)
    gt_masks = (gt_masks > 0.5).astype(np.float32)

    ious = iou_binary(pred_masks, gt_masks, eps=eps)
    
    return ious

def iou_localization_accuracy_from_cams(
    cams,
    masks,
    top_k_percent = 0.15,
    iou_thresholds = (0.3, 0.5),
    eps = 1e-8,
):
    """
    Compute localization accuracy based on IoU thresholds of input CAMs.

    Args:
        cams: Class Activation Maps, shape (B, H, W).
        masks: Ground truth binary masks, shape (B, H, W).
        top_k_percent: Percentage of top pixels in CAM to consider as predicted mask.
        iou_thresholds: Tuple of IoU thresholds for accuracy computation.
            Should be in the range (0, 1).
        eps: Small value to avoid division by zero.
    
    Returns:
        accuracies: Dictionary mapping IoU threshold to accuracy.
    
    Notes:
        This function just combines IoU computation and accuracy calculation.
    """
    ious = ious_from_cams(
        cams,
        masks,
        top_k_percent=top_k_percent,
        eps=eps,
    )

    accuracies = {}
    for thresh in iou_thresholds:
        correct = (ious >= thresh).sum()
        accuracy = correct / len(ious)
        accuracies[thresh] = accuracy

    return accuracies

def iou_localization_accuracy(
    ious,
    iou_thresholds = (0.3, 0.5),
):
    """
    Compute localization accuracy of given IoU scores.
    Args:
        ious: Array of IoU scores, shape (B,).
        iou_thresholds: Tuple of IoU thresholds for accuracy computation.
            Should be in the range (0, 1).
    Returns:
        accuracies: Dictionary mapping IoU threshold to accuracy.
    """
    if not isinstance(ious, np.ndarray):
        try:
            ious = np.array(ious)
        except:
            raise TypeError(f"Expected ious to be a numpy array or convertible to one, got {type(ious)} instead.")

    accuracies = {}
    for thresh in iou_thresholds:
        correct = (ious >= thresh).sum()
        accuracy = correct / len(ious)
        accuracies[thresh] = accuracy

    return accuracies


# Example usage:

if __name__ == "__main__":
    # Dummy CAMs and masks
    cams = np.random.rand(5, 224, 224).astype(np.float32)
    masks = (np.random.rand(5, 224, 224) > 0.5).astype(np.uint8)

    binarized_cams = binarize_heatmap(cams, top_k_percent=0.15)
    print("Binarized CAMs shape:", binarized_cams.shape)
    print("Binarized CAMs sample:", binarized_cams[0])

    ious = ious_from_cams(
        cams,
        masks,
        top_k_percent=0.15
    )

    accuracies = iou_localization_accuracy(
        ious,
        iou_thresholds=(0.3, 0.5),
    )

    print("IoUs:", ious)
    print("Localization Accuracies:", accuracies)