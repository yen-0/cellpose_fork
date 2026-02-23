import numpy as np
from .interpolation import interpolate_missing_mask
from .confidence import track_confidence


def _node_mask(node, shape, slice_mask=None):
    if hasattr(node, "mask"):
        return node.mask
    if hasattr(node, "full_mask"):
        return node.full_mask(shape, slice_mask=slice_mask)
    raise AttributeError("node does not provide mask/full_mask")


def _select_best_unoccupied_structure(pred_mask, slice_mask, occupied_mask, max_centroid_dist=40.0):
    ids = np.unique(slice_mask)
    ids = ids[ids > 0]
    if ids.size == 0:
        return None, 0.0

    ys, xs = np.where(pred_mask)
    if ys.size == 0:
        return None, 0.0
    pred_centroid = np.array([ys.mean(), xs.mean()], dtype=np.float32)
    pred_area = float(pred_mask.sum())

    best_mask, best_score = None, -1e9
    for inst_id in ids:
        cand = slice_mask == inst_id
        if not np.any(np.logical_and(cand, np.logical_not(occupied_mask))):
            continue
        cys, cxs = np.where(cand)
        ccent = np.array([cys.mean(), cxs.mean()], dtype=np.float32)
        dist = float(np.linalg.norm(ccent - pred_centroid))
        if dist > max_centroid_dist:
            continue
        inter = float(np.logical_and(pred_mask, cand).sum())
        union = float(np.logical_or(pred_mask, cand).sum())
        iou = inter / (union + 1e-6)
        ar = min(pred_area, float(cand.sum())) / (max(pred_area, float(cand.sum())) + 1e-6)
        score = iou + 0.4 * ar - 0.01 * dist
        if score > best_score:
            best_score = score
            best_mask = cand

    return best_mask, float(best_score if best_mask is not None else 0.0)


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

    # edge-fill now uses nearest observed mask as prediction prior only
    if fill_edges and nodes:
        first, last = nodes[0], nodes[-1]
        first_src = None if source_masks is None else source_masks[first.z]
        last_src = None if source_masks is None else source_masks[last.z]

        first_prior = _node_mask(first, image_stack[0].shape[:2], slice_mask=first_src)
        for z in range(0, first.z):
            recoveries.append((z, first_prior, 0.5, True))

        last_prior = _node_mask(last, image_stack[0].shape[:2], slice_mask=last_src)
        for z in range(last.z + 1, len(image_stack)):
            recoveries.append((z, last_prior, 0.5, True))

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
        has_first_slice_node = any(n.z == 0 for n in tr.nodes)
        if not has_first_slice_node and (len(tr.nodes) < min_track_len or conf < min_conf):
            continue
        kept.append((tr, conf))

    kept = sorted(kept, key=lambda x: x[1], reverse=True)

    locked_direct = [np.zeros(image_stack[0].shape[:2], dtype=bool) for _ in range(zcount)]
    track_to_tid = {}
    for tr, _ in kept:
        tid = next_id
        next_id += 1
        track_to_tid[id(tr)] = tid
        for n in tr.nodes:
            src = None if source_masks is None else source_masks[n.z]
            nmask = _node_mask(n, image_stack[0].shape[:2], slice_mask=src)
            force_keep_first_slice = (n.z == 0)
            nmask_use = nmask if (force_keep_first_slice or not avoid_occupied) else np.logical_and(nmask, out[n.z] == 0)
            if not np.any(nmask_use):
                continue
            out[n.z][nmask_use] = tid
            locked_direct[n.z][nmask_use] = True
            if track_labels is not None:
                track_labels[n.z][nmask_use] = tid

    for tr, _ in kept:
        tid = track_to_tid[id(tr)]
        for z, m_pred, _, _ in recover_track_gaps(tr, image_stack, flow_stack=flow_stack, fill_edges=fill_edges, source_masks=source_masks):
            if m_pred is None or not np.any(m_pred):
                continue
            if source_masks is None:
                continue

            free = out[z] == 0
            not_locked = np.logical_not(locked_direct[z])
            occ = np.logical_and(free, not_locked)
            if prob_stack is not None:
                occ = np.logical_and(occ, prob_stack[z] < prob_occupancy_thresh)

            chosen, _ = _select_best_unoccupied_structure(m_pred, source_masks[z], occupied_mask=np.logical_not(occ))
            if chosen is None:
                continue

            m_use = np.logical_and(chosen, occ)
            if avoid_occupied:
                frac = m_use.sum() / (chosen.sum() + 1e-6)
                if frac < min_free_fraction:
                    continue
            if not np.any(m_use):
                continue

            out[z][m_use] = tid
            if track_labels is not None:
                track_labels[z][m_use] = tid
            reconstructed_flags[z][m_use] = 1

    return out, track_labels, reconstructed_flags, kept
