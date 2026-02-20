import numpy as np
from .interpolation import interpolate_missing_mask
from .confidence import track_confidence


def recover_track_gaps(track, image_stack, flow_stack=None):
    """Mandatory reconstruction for all missing slices inside valid linked tracks."""
    recoveries = []
    nodes = sorted(track.nodes, key=lambda n: n.z)
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if b.z - a.z > 1:
            for z in range(a.z + 1, b.z):
                flow_slice = None if flow_stack is None else flow_stack[z]
                mask_hat, conf = interpolate_missing_mask(a, b, z, image_stack[z], flow_slice=flow_slice)
                recoveries.append((z, mask_hat, conf, False))
    return recoveries


def relabel_tracks(tracks, image_stack, flow_stack=None, min_track_len=2, min_conf=0.05):
    zcount = len(image_stack)
    out = [np.zeros(image_stack[0].shape[:2], dtype=np.int32) for _ in range(zcount)]
    track_labels = np.zeros((zcount, *image_stack[0].shape[:2]), dtype=np.int32)
    reconstructed_flags = np.zeros((zcount, *image_stack[0].shape[:2]), dtype=np.uint8)
    next_id = 1
    kept = []

    for tr in tracks:
        conf = track_confidence(tr, image_stack)
        if len(tr.nodes) < min_track_len or conf < min_conf:
            continue
        kept.append((tr, conf))

    for tr, _ in kept:
        tid = next_id
        next_id += 1
        for n in tr.nodes:
            out[n.z][n.mask] = tid
            track_labels[n.z][n.mask] = tid
        for z, m, _, _ in recover_track_gaps(tr, image_stack, flow_stack=flow_stack):
            if m is not None and np.any(m):
                out[z][m] = tid
                track_labels[z][m] = tid
                reconstructed_flags[z][m] = 1

    return out, track_labels, reconstructed_flags, kept
