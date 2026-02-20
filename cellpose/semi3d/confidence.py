import numpy as np


def track_confidence(track, image_stack):
    length_score = min(1.0, len(track.nodes) / 4.0)
    mean_link_iou = float(np.mean(track.links)) if track.links else 0.5
    areas = np.array([n.area for n in track.nodes], dtype=np.float32)
    shape_consistency = 1.0 - min(1.0, float(np.std(areas) / (np.mean(areas) + 1e-6))) if len(areas) > 1 else 0.6

    intensity_vals = []
    for n in track.nodes:
        im = image_stack[n.z]
        if im.ndim == 3:
            im = im.mean(axis=-1)
        if np.any(n.mask):
            intensity_vals.append(float(im[n.mask].mean()))
    intensity_support = float(np.clip(np.mean(intensity_vals) / (np.max(image_stack) + 1e-6), 0.0, 1.0)) if intensity_vals else 0.5
    gap_penalty = max(0.0, 1.0 - 0.2 * track.gap_bridges)

    return length_score * mean_link_iou * shape_consistency * intensity_support * gap_penalty
