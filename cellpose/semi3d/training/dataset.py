import os
from cellpose import io


def _find_gt_path(input_dir, stem, ext, gt_suffix="_masks"):
    tif_gt = os.path.join(input_dir, f"{stem}{gt_suffix}{ext}")
    if os.path.exists(tif_gt):
        return tif_gt
    seg_npy = os.path.join(input_dir, f"{stem}_seg.npy")
    if os.path.exists(seg_npy):
        return seg_npy
    return None


def load_stacks(input_dir, gt_suffix="_masks"):
    """Load training stacks with paired GT masks from .tif or _seg.npy annotations."""
    images, labels = [], []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        if gt_suffix in name:
            continue
        stem, ext = os.path.splitext(name)
        gt_path = _find_gt_path(input_dir, stem, ext, gt_suffix=gt_suffix)
        if gt_path is None:
            continue
        img_path = os.path.join(input_dir, name)
        images.append(io.imread(img_path))
        labels.append(io.imread(gt_path))
    return images, labels
