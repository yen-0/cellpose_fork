import numpy as np
from .interpolation import interpolate_missing_mask, refine_mask_with_slice
from .confidence import track_confidence


def _node_mask(node, shape, slice_mask=None):
    if hasattr(node, "mask"):
        return node.mask
    if hasattr(node, "full_mask"):
        return node.full_mask(shape, slice_mask=slice_mask)
    raise AttributeError("node does not provide mask/full_mask")


def recover_track_gaps(track, image_stack, flow_stack=None, fill_edges=False, source_masks=None):
    recoveries = []
    nodes = sorted(track.nodes, key=lambda n: n.z)

    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if b.z - a.z > 1:
            for z in range(a.z + 1, b.z):
                flow_slice = None if flow_stack is None else flow_stack[z]
                pa = None if source_masks is None else source_masks[a.z]
                pb = None if source_masks is None else source_masks[b.z]
                mask_hat, conf = interpolate_missing_mask(
                    a, b, z, image_stack[z], flow_slice=flow_slice, prev_slice_mask=pa, next_slice_mask=pb
                )
                recoveries.append((z, mask_hat, conf, False))

    if fill_edges and nodes:
        first, last = nodes[0], nodes[-1]
        first_src = None if source_masks is None else source_masks[first.z]
        last_src = None if source_masks is None else source_masks[last.z]

        # Run reconstruction-like refinement on boundary slices using nearest observed mask as prior.
        first_prior = _node_mask(first, image_stack[0].shape[:2], slice_mask=first_src)
        for z in range(0, first.z):
            flow_slice = None if flow_stack is None else flow_stack[z]
            mask_hat = refine_mask_with_slice(first_prior, image_stack[z], flow_slice=flow_slice)
            recoveries.append((z, mask_hat, 0.5, True))

        last_prior = _node_mask(last, image_stack[0].shape[:2], slice_mask=last_src)
        for z in range(last.z + 1, len(image_stack)):
            flow_slice = None if flow_stack is None else flow_stack[z]
            mask_hat = refine_mask_with_slice(last_prior, image_stack[z], flow_slice=flow_slice)
            recoveries.append((z, mask_hat, 0.5, True))

    return recoveries


def relabel_tracks(tracks, image_stack, flow_stack=None, min_track_len=2, min_conf=0.05,
                   fill_edges=False, return_track_labels=True, source_masks=None, avoid_occupied=True, min_free_fraction=0.25,
                   prob_stack=None, prob_occupancy_thresh=0.5):
    zcount = len(image_stack)
    out = [np.zeros(image_stack[0].shape[:2], dtype=np.int32) for _ in range(zcount)]
    track_labels = np.zeros((zcount, *image_stack[0].shape[:2]), dtype=np.int32) if return_track_labels else None
    reconstructed_flags = np.zeros((zcount, *image_stack[0].shape[:2]), dtype=np.uint8)
    next_id = 1
    kept = []

    for tr in tracks:
        conf = track_confidence(tr, image_stack, source_masks=source_masks)
        if len(tr.nodes) < min_track_len or conf < min_conf:
            continue
        kept.append((tr, conf))

    # highest-confidence linked tracks claim territory first
    kept = sorted(kept, key=lambda x: x[1], reverse=True)

    # pass 1: place only directly linked node masks and lock them
    locked_direct = [np.zeros(image_stack[0].shape[:2], dtype=bool) for _ in range(zcount)]
    track_to_tid = {}
    for tr, _ in kept:
        tid = next_id
        next_id += 1
        track_to_tid[id(tr)] = tid
        for n in tr.nodes:
            src = None if source_masks is None else source_masks[n.z]
            nmask = _node_mask(n, image_stack[0].shape[:2], slice_mask=src)
            nmask_use = nmask
            if avoid_occupied:
                free = out[n.z] == 0
                nmask_use = np.logical_and(nmask, free)
            if not np.any(nmask_use):
                continue
            out[n.z][nmask_use] = tid
            locked_direct[n.z][nmask_use] = True
            if track_labels is not None:
                track_labels[n.z][nmask_use] = tid

    # pass 2: reconstruction strictly limited to non-occupied / non-locked territory
    for tr, _ in kept:
        tid = track_to_tid[id(tr)]
        for z, m, _, _ in recover_track_gaps(tr, image_stack, flow_stack=flow_stack, fill_edges=fill_edges, source_masks=source_masks):
            if m is None or not np.any(m):
                continue

            m_use = m
            # reconstruction can only happen in NOT OCCUPIED TERRITORIES
            free = out[z] == 0
            not_locked = np.logical_not(locked_direct[z])
            m_use = np.logical_and(m_use, np.logical_and(free, not_locked))
            # optional probability occupancy prior from stage1 cellprob
            if prob_stack is not None:
                pz = prob_stack[z]
                m_use = np.logical_and(m_use, pz < prob_occupancy_thresh)

            if avoid_occupied:
                frac = m_use.sum() / (m.sum() + 1e-6)
                if frac < min_free_fraction:
                    continue

            if not np.any(m_use):
                continue
            out[z][m_use] = tid
            if track_labels is not None:
                track_labels[z][m_use] = tid
            reconstructed_flags[z][m_use] = 1

    return out, track_labels, reconstructed_flags, kept
