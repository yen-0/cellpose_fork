import os
import numpy as np
from cellpose import io


def load_stacks(input_dir, gt_suffix="_masks"):
    images, labels = [], []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        if gt_suffix in name:
            continue
        img_path = os.path.join(input_dir, name)
        stem, ext = os.path.splitext(name)
        gt_path = os.path.join(input_dir, f"{stem}{gt_suffix}{ext}")
        if os.path.exists(gt_path):
            images.append(io.imread(img_path))
            labels.append(io.imread(gt_path))
    return images, labels
