import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable

from .interpolation import interpolate_missing_mask
from .confidence import track_confidence


def _node_mask(node, shape, slice_mask=None):
    if hasattr(node, "mask"):
        return node.mask
    if hasattr(node, "full_mask"):
        return node.full_mask(shape, slice_mask=slice_mask)
    raise AttributeError("node does not provide mask/full_mask")




def _build_slice_instance_cache(slice_mask):
    ids = np.unique(slice_mask)
    ids = ids[ids > 0]
    if ids.size == 0:
        return {"ids": np.zeros((0,), dtype=np.int32), "masks": [], "centroids": np.zeros((0, 2), dtype=np.float32),
                "areas": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.int32)}

    masks = []
    centroids = []
    areas = []
    bboxes = []
    ids_out = []
    for inst_id in ids.tolist():
        cand = (slice_mask == inst_id)
        ys, xs = np.where(cand)
        if ys.size == 0:
            continue
        masks.append(cand)
        centroids.append((float(ys.mean()), float(xs.mean())))
        areas.append(float(ys.size))
        bboxes.append((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1))
        ids_out.append(int(inst_id))

    return {
        "ids": np.asarray(ids_out, dtype=np.int32),
        "masks": masks,
        "centroids": np.asarray(centroids, dtype=np.float32),
        "areas": np.asarray(areas, dtype=np.float32),
        "bboxes": np.asarray(bboxes, dtype=np.int32),
    }
def _select_best_unoccupied_structure(pred_mask, slice_mask, occupied_mask, max_centroid_dist=40.0, slice_cache=None):
    if slice_cache is None:
        slice_cache = _build_slice_instance_cache(slice_mask)

    if slice_cache["ids"].size == 0:
        return None, 0.0

    ys, xs = np.where(pred_mask)
    if ys.size == 0:
        return None, 0.0
    pred_centroid = np.array([ys.mean(), xs.mean()], dtype=np.float32)
    pred_area = float(pred_mask.sum())
    py0, py1, px0, px1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    best_mask, best_score = None, -1e9
    cand_cent = slice_cache["centroids"]
    if cand_cent.size == 0:
        return None, 0.0

    dists = np.linalg.norm(cand_cent - pred_centroid[None, :], axis=1)
    bbox = slice_cache["bboxes"]
    pad = 8
    overlaps_bbox = (bbox[:, 0] < (py1 + pad)) & (bbox[:, 1] > (py0 - pad)) & (bbox[:, 2] < (px1 + pad)) & (bbox[:, 3] > (px0 - pad))
    candidate_idx = np.where((dists <= float(max_centroid_dist)) & overlaps_bbox)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.where(dists <= float(max_centroid_dist))[0]

    for idx in candidate_idx.tolist():
        cand = slice_cache["masks"][idx]
        if not np.any(np.logical_and(cand, np.logical_not(occupied_mask))):
            continue
        dist = float(dists[idx])
        inter = float(np.logical_and(pred_mask, cand).sum())
        union = float(np.logical_or(pred_mask, cand).sum())
        iou = inter / (union + 1e-6)
        carea = float(slice_cache["areas"][idx])
        ar = min(pred_area, carea) / (max(pred_area, carea) + 1e-6)
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
                   prob_stack=None, prob_occupancy_thresh=0.5, show_progress=False,
                   skip_gap_reconstruction=False, recon_workers=1):
    zcount = len(image_stack)
    out = [np.zeros(image_stack[0].shape[:2], dtype=np.int32) for _ in range(zcount)]
    track_labels = np.zeros((zcount, *image_stack[0].shape[:2]), dtype=np.int32) if return_track_labels else None
    reconstructed_flags = np.zeros((zcount, *image_stack[0].shape[:2]), dtype=np.uint8)
    next_id = 1
    kept = []

    image_max = float(np.max(image_stack))
    gray_stack = [im.mean(axis=-1) if getattr(im, "ndim", 2) == 3 else im for im in image_stack]

    track_iter = tqdm(tracks, desc="[semi3d:stage2] filtering tracks", unit="track") if show_progress else tracks
    for tr in track_iter:
        conf = track_confidence(
            tr,
            image_stack,
            source_masks=source_masks,
            image_max=image_max,
            gray_stack=gray_stack,
        )
        has_first_slice_node = any(n.z == 0 for n in tr.nodes)
        if not has_first_slice_node and (len(tr.nodes) < min_track_len or conf < min_conf):
            continue
        kept.append((tr, conf))

    kept = sorted(kept, key=lambda x: x[1], reverse=True)

    slice_caches = None
    if source_masks is not None:
        slice_caches = [_build_slice_instance_cache(source_masks[z]) for z in range(zcount)]

    locked_direct = [np.zeros(image_stack[0].shape[:2], dtype=bool) for _ in range(zcount)]
    track_to_tid = {}
    direct_iter = tqdm(kept, desc="[semi3d:stage2] placing direct masks", unit="track") if show_progress else kept
    for tr, _ in direct_iter:
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

    if skip_gap_reconstruction:
        return out, track_labels, reconstructed_flags, kept

    worker_count = int(recon_workers) if recon_workers is not None else 1
    if worker_count <= 0:
        worker_count = os.cpu_count() or 1

    def _recover_for_track(entry):
        tr, _ = entry
        rec = recover_track_gaps(tr, image_stack, flow_stack=flow_stack, fill_edges=fill_edges, source_masks=source_masks)
        return id(tr), rec

    if worker_count > 1 and len(kept) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            recovered = dict(ex.map(_recover_for_track, kept))
    else:
        recovered = {id(tr): recover_track_gaps(tr, image_stack, flow_stack=flow_stack, fill_edges=fill_edges, source_masks=source_masks)
                    for tr, _ in kept}

    recon_iter = tqdm(kept, desc="[semi3d:stage2] reconstructing gaps", unit="track") if show_progress else kept
    for tr, _ in recon_iter:
        tid = track_to_tid[id(tr)]
        for z, m_pred, _, _ in recovered.get(id(tr), []):
            if m_pred is None or not np.any(m_pred):
                continue
            if source_masks is None:
                continue

            free = out[z] == 0
            not_locked = np.logical_not(locked_direct[z])
            occ = np.logical_and(free, not_locked)
            if prob_stack is not None:
                occ = np.logical_and(occ, prob_stack[z] < prob_occupancy_thresh)

            chosen, _ = _select_best_unoccupied_structure(m_pred, source_masks[z], occupied_mask=np.logical_not(occ), slice_cache=None if slice_caches is None else slice_caches[z])
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
