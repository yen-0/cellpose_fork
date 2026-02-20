import numpy as np


def dice_score(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom else 1.0


def iou_score(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter / union) if union else 1.0


def z_continuity_f1(mask_stack):
    present = np.array([np.any(m > 0) for m in mask_stack])
    tp = np.sum(present[1:] & present[:-1])
    fp = np.sum(present[1:] & ~present[:-1])
    fn = np.sum(~present[1:] & present[:-1])
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    return 2 * p * r / (p + r + 1e-6)


def false_split_rate(mask_stack):
    splits = 0
    valid = 0
    for z in range(1, len(mask_stack)):
        prev = len(np.unique(mask_stack[z - 1])) - 1
        cur = len(np.unique(mask_stack[z])) - 1
        if prev > 0:
            valid += 1
            if cur > prev:
                splits += 1
    return splits / valid if valid else 0.0


def recovery_iou(gt_slice, recovered_slice):
    return iou_score(gt_slice, recovered_slice)
