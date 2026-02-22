import numpy as np


def _node_mask(node, shape, slice_mask=None):
    if hasattr(node, "mask"):
        return node.mask
    if hasattr(node, "full_mask"):
        return node.full_mask(shape, slice_mask=slice_mask)
    raise AttributeError("node does not provide mask/full_mask")


def track_confidence(track, image_stack, source_masks=None):
    length_score = min(1.0, len(track.nodes) / 4.0)
    mean_link_iou = float(np.mean(track.links)) if track.links else 0.5
    areas = np.array([n.area for n in track.nodes], dtype=np.float32)
    shape_consistency = 1.0 - min(1.0, float(np.std(areas) / (np.mean(areas) + 1e-6))) if len(areas) > 1 else 0.6

    intensity_vals = []
    for n in track.nodes:
        im = image_stack[n.z]
        if im.ndim == 3:
            im = im.mean(axis=-1)
        src = None if source_masks is None else source_masks[n.z]
        nmask = _node_mask(n, image_stack[0].shape[:2], slice_mask=src)
        if np.any(nmask):
            intensity_vals.append(float(im[nmask].mean()))
    intensity_support = float(np.clip(np.mean(intensity_vals) / (np.max(image_stack) + 1e-6), 0.0, 1.0)) if intensity_vals else 0.5

    gap_hist = getattr(track, "gap_hist", {})
    gap1 = gap_hist.get(1, 0)
    gap2 = gap_hist.get(2, 0)
    long_gap = sum(v for k, v in gap_hist.items() if k >= 3)
    gap_penalty = max(0.2, 1.0 - 0.12 * gap1 - 0.18 * gap2 - 0.25 * long_gap)

    return length_score * mean_link_iou * shape_consistency * intensity_support * gap_penalty
