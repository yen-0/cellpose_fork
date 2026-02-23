from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2


@dataclass
class InstanceNode:
    z: int
    instance_id: int
    bbox: Tuple[int, int, int, int]  # y0, y1, x0, x1
    centroid: np.ndarray
    area: int
    mask_crop: Optional[np.ndarray] = None
    contours: Optional[List[np.ndarray]] = None  # polygon approximation (local crop coords)

    def compact(self):
        self.mask_crop = None

    def full_mask(self, shape: Tuple[int, int], slice_mask: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        y0, y1, x0, x1 = self.bbox
        if self.mask_crop is not None:
            out[y0:y1, x0:x1] = self.mask_crop
            return out
        if slice_mask is None:
            raise ValueError("slice_mask required when node mask_crop was compacted")
        out[y0:y1, x0:x1] = (slice_mask[y0:y1, x0:x1] == self.instance_id)
        return out


@dataclass
class Track:
    track_id: int
    nodes: List[InstanceNode] = field(default_factory=list)
    links: List[float] = field(default_factory=list)
    gap_bridges: int = 0
    gap_hist: Dict[int, int] = field(default_factory=dict)


def _extract_contours(mask_crop: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        eps = 0.01 * cv2.arcLength(c, True)
        out.append(cv2.approxPolyDP(c, eps, True))
    return out


def _node_mask_crop(node: InstanceNode, slice_mask: Optional[np.ndarray] = None):
    if node.mask_crop is not None:
        return node.mask_crop
    if slice_mask is None:
        raise ValueError("slice_mask required for compacted node")
    y0, y1, x0, x1 = node.bbox
    return (slice_mask[y0:y1, x0:x1] == node.instance_id)


def _node_iou(node_a: InstanceNode, node_b: InstanceNode, slice_mask_a: Optional[np.ndarray] = None, slice_mask_b: Optional[np.ndarray] = None) -> float:
    ay0, ay1, ax0, ax1 = node_a.bbox
    by0, by1, bx0, bx1 = node_b.bbox
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return 0.0

    a_crop = _node_mask_crop(node_a, slice_mask_a)
    b_crop = _node_mask_crop(node_b, slice_mask_b)
    a_view = a_crop[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
    b_view = b_crop[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    inter = np.logical_and(a_view, b_view).sum()
    union = node_a.area + node_b.area - inter
    return float(inter / union) if union > 0 else 0.0


def _node_overlap_pair(node_prev: InstanceNode, node_cur: InstanceNode):
    """Return (IoU, overlap_prev, overlap_cur), where overlap_prev uses prev area denominator."""
    ay0, ay1, ax0, ax1 = node_prev.bbox
    by0, by1, bx0, bx1 = node_cur.bbox
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return 0.0, 0.0, 0.0

    a_crop = _node_mask_crop(node_prev)
    b_crop = _node_mask_crop(node_cur)
    a_view = a_crop[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
    b_view = b_crop[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    inter = float(np.logical_and(a_view, b_view).sum())
    union = float(node_prev.area + node_cur.area - inter)
    iou = inter / (union + 1e-6)
    overlap_prev = inter / (float(node_prev.area) + 1e-6)
    overlap_cur = inter / (float(node_cur.area) + 1e-6)
    return float(iou), float(overlap_prev), float(overlap_cur)


def _polygon_similarity(node_a: InstanceNode, node_b: InstanceNode) -> float:
    if not node_a.contours or not node_b.contours:
        return 0.0
    c1 = max(node_a.contours, key=cv2.contourArea)
    c2 = max(node_b.contours, key=cv2.contourArea)
    try:
        d = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
        return float(1.0 / (1.0 + d))
    except Exception:
        return 0.0


def _merge_nodes(nodes: List[InstanceNode]) -> InstanceNode:
    z = nodes[0].z
    ids = [n.instance_id for n in nodes]
    y0 = min(n.bbox[0] for n in nodes)
    y1 = max(n.bbox[1] for n in nodes)
    x0 = min(n.bbox[2] for n in nodes)
    x1 = max(n.bbox[3] for n in nodes)
    merged = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for n in nodes:
        ny0, ny1, nx0, nx1 = n.bbox
        m = _node_mask_crop(n)
        merged[ny0 - y0:ny1 - y0, nx0 - x0:nx1 - x0] |= m
    ys, xs = np.where(merged)
    centroid = np.array([ys.mean() + y0, xs.mean() + x0], dtype=np.float32) if ys.size else np.array([0.0, 0.0], dtype=np.float32)
    return InstanceNode(
        z=z,
        instance_id=int(min(ids)),
        bbox=(y0, y1, x0, x1),
        centroid=centroid,
        area=int(merged.sum()),
        mask_crop=merged,
        contours=_extract_contours(merged),
    )


def _extract_nodes_for_slice(slice_mask: np.ndarray, z: int) -> List[InstanceNode]:
    nodes: List[InstanceNode] = []
    ids = np.unique(slice_mask)
    ids = ids[ids > 0]
    for inst_id in ids:
        ys, xs = np.where(slice_mask == inst_id)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = (slice_mask[y0:y1, x0:x1] == inst_id)
        centroid = np.array([ys.mean(), xs.mean()], dtype=np.float32)
        nodes.append(InstanceNode(z=z, instance_id=int(inst_id), bbox=(y0, y1, x0, x1), centroid=centroid,
                                  area=int(crop.sum()), mask_crop=crop, contours=_extract_contours(crop)))
    return nodes


def _build_spatial_grid(nodes: List[InstanceNode], cell_size: float):
    grid = {}
    cs = max(1.0, float(cell_size))
    for i, n in enumerate(nodes):
        gy = int(n.centroid[0] // cs)
        gx = int(n.centroid[1] // cs)
        grid.setdefault((gy, gx), []).append(i)
    return grid, cs


def _query_neighbors(node: InstanceNode, grid, cell_size: float):
    gy = int(node.centroid[0] // cell_size)
    gx = int(node.centroid[1] // cell_size)
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out.extend(grid.get((gy + dy, gx + dx), []))
    return out


def _gpu_prefilter_candidates(active, current_nodes, link_dist, size_tolerance, grid=None, cell_size=None, topk=16):
    """GPU-accelerated local candidate proposal around each track (not full-field all-pairs)."""
    try:
        import torch
        if not torch.cuda.is_available() or len(active) == 0 or len(current_nodes) == 0:
            return None

        c_cent = np.stack([n.centroid for n in current_nodes]).astype(np.float32)
        c_area = np.array([n.area for n in current_nodes], dtype=np.float32)
        C = torch.from_numpy(c_cent).to("cuda")
        CA = torch.from_numpy(c_area).to("cuda")

        out = {}
        max_dist = float(link_dist) * 1.4
        min_area = float(size_tolerance) * 0.35

        for ai, tr in enumerate(active):
            last = tr.nodes[-1]
            if grid is not None and cell_size is not None:
                neighbor_idx = _query_neighbors(last, grid, cell_size)
            else:
                neighbor_idx = list(range(len(current_nodes)))
            if not neighbor_idx:
                out[ai] = []
                continue

            # de-duplicate while preserving order
            neighbor_idx = list(dict.fromkeys(int(v) for v in neighbor_idx))
            idx_t = torch.tensor(neighbor_idx, dtype=torch.long, device="cuda")
            CC = C.index_select(0, idx_t)
            CA_sub = CA.index_select(0, idx_t)

            a_cent = torch.tensor(last.centroid, dtype=torch.float32, device="cuda").unsqueeze(0)
            dists = torch.cdist(a_cent, CC)[0]
            a_area = torch.tensor(float(last.area), dtype=torch.float32, device="cuda")
            area_ratio = torch.minimum(a_area, CA_sub) / torch.maximum(a_area, CA_sub)

            valid = (dists <= max_dist) & (area_ratio >= min_area)
            score = (1.0 - dists / (max_dist + 1e-6)) + 0.6 * area_ratio
            score = torch.where(valid, score, torch.full_like(score, -1e9))

            k = int(max(1, min(topk, score.shape[0])))
            vals, idx = torch.topk(score, k=k)
            vals_cpu = vals.detach().cpu().numpy()
            idx_cpu = idx.detach().cpu().numpy()
            keep = [neighbor_idx[int(j)] for j, v in zip(idx_cpu, vals_cpu) if v > -1e8]
            out[ai] = keep

        return out
    except Exception:
        return None


def _track_reference_node(track: Track) -> InstanceNode:
    """Use a short history to avoid over-shrunk previous mask when scoring next slice."""
    last = track.nodes[-1]
    if len(track.nodes) < 2:
        return last
    prev = track.nodes[-2]
    # if current shrank strongly, blend with previous support mask
    if last.area < 0.88 * prev.area:
        ref = _merge_nodes([prev, last])
        ref.z = last.z
        return ref
    return last


def _collect_track_candidates(ai, tr, z, current_nodes, used, candidate_idx, link_iou,
                              link_dist, size_tolerance, max_gap):
    last = tr.nodes[-1]
    if z - last.z > max_gap:
        return []

    ref = _track_reference_node(tr)
    cand = []
    for i in candidate_idx:
        if i in used:
            continue
        node = current_nodes[i]
        gap = node.z - last.z
        if gap < 1 or gap > max_gap:
            continue
        dist = float(np.linalg.norm(node.centroid - last.centroid))
        area_ratio = float(min(ref.area, node.area) / max(ref.area, node.area))
        cand.append((i, node, gap, dist, area_ratio))
    if not cand:
        return []

    sparse_pool = len(cand) <= 1
    closest_idx = min(cand, key=lambda t: t[3])[0]
    edges = []
    for i, node, gap, dist, area_ratio in cand:
        dyn_dist = link_dist * (1.45 + min(2.2, 0.95 * (gap - 1)))
        if dist > dyn_dist and i != closest_idx:
            continue

        # area stays relative and soft, especially for closest candidate.
        area_floor = size_tolerance * 0.18 * (1.0 - min(0.55, 0.18 * (gap - 1)))
        if sparse_pool:
            area_floor *= 0.25
        if area_ratio < area_floor and i != closest_idx:
            continue

        iou, ov_prev, ov_cur = _node_overlap_pair(ref, node)
        poly_sim = _polygon_similarity(ref, node)
        weak_link = (ov_prev < (link_iou * 0.12) and ov_cur < 0.20 and gap == 1 and poly_sim < 0.06)
        if weak_link and not sparse_pool and i != closest_idx:
            continue

        closest_bonus = 0.24 if i == closest_idx else 0.0
        sparse_bonus = 0.20 if sparse_pool and weak_link else 0.0
        score = (
            0.70 * ov_prev
            + 0.20 * ov_cur
            + 0.30 * area_ratio
            + 0.10 * poly_sim
            - 0.0045 * dist
            - 0.008 * (gap - 1)
            + closest_bonus
            + sparse_bonus
        )
        edges.append((score, ai, i, iou, gap))
    return edges


def _compact_track_history(active: List[Track], keep_recent: int = 2):
    # Disabled: compaction can invalidate later overlap/merge computations that need mask crops.
    # Keep full crops to avoid "slice_mask required for compacted node" runtime failures.
    return


def build_association_tracks(slice_masks, link_iou=0.1, link_dist=30.0, size_tolerance=0.6,
                             max_gap=3, gpu_prefilter=False, merge_dist=12.0, link_workers=1,
                             return_debug=False, force_attach_min_area=8, anchor_first_slice=True):
    tracks: List[Track] = []
    active: List[Track] = []
    next_track_id = 1
    debug_records = []

    for z in range(len(slice_masks)):
        current_nodes = _extract_nodes_for_slice(slice_masks[z], z)
        used = set()

        node_debug = []
        for i, n in enumerate(current_nodes):
            node_debug.append({
                "node_index": int(i),
                "instance_id": int(n.instance_id),
                "centroid": [float(n.centroid[0]), float(n.centroid[1])],
                "track_id": None,
                "status": "unprocessed",
                "reason": "",
                "score": None,
            })

        if z == 0:
            for i, node in enumerate(current_nodes):
                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)
                used.add(i)
                node_debug[i].update({
                    "track_id": int(tr.track_id),
                    "status": "anchor",
                    "reason": "initialized_from_first_slice",
                    "score": None,
                })
            if return_debug:
                debug_records.append({
                    "z": int(z),
                    "nodes": node_debug,
                    "tracks": [{"track_id": int(t.track_id), "last_z": int(t.nodes[-1].z), "status": "anchor"} for t in active],
                })
            continue

        grid, cs = _build_spatial_grid(current_nodes, link_dist)
        gpu_candidates = _gpu_prefilter_candidates(active, current_nodes, link_dist, size_tolerance, grid=grid, cell_size=cs) if gpu_prefilter else None

        per_track_candidates = []
        for ai, tr in enumerate(active):
            last = tr.nodes[-1]
            if z - last.z > max_gap:
                per_track_candidates.append((ai, tr, []))
                continue
            candidate_idx = gpu_candidates.get(ai, []) if gpu_candidates is not None else _query_neighbors(last, grid, cs)
            per_track_candidates.append((ai, tr, candidate_idx))

        all_edges = []
        worker_count = int(max(1, link_workers))
        if worker_count > 1 and len(per_track_candidates) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futs = [
                    ex.submit(
                        _collect_track_candidates,
                        ai,
                        tr,
                        z,
                        current_nodes,
                        used,
                        candidate_idx,
                        link_iou,
                        link_dist,
                        size_tolerance,
                        max_gap,
                    )
                    for ai, tr, candidate_idx in per_track_candidates
                ]
                for f in futs:
                    all_edges.extend(f.result())
        else:
            for ai, tr, candidate_idx in per_track_candidates:
                all_edges.extend(_collect_track_candidates(
                    ai, tr, z, current_nodes, used, candidate_idx,
                    link_iou, link_dist, size_tolerance, max_gap,
                ))

        all_edges.sort(key=lambda e: e[0], reverse=True)
        candidate_nodes = {idx for _, _, idx, _, _ in all_edges}
        best_score_by_node = {}
        for sc, _, idx, _, _ in all_edges:
            if idx not in best_score_by_node or sc > best_score_by_node[idx]:
                best_score_by_node[idx] = float(sc)

        claimed_tracks = set()
        assigned_track_ai = {}
        for score, ai, idx, iou, gap in all_edges:
            if ai in claimed_tracks or idx in used:
                continue
            tr = active[ai]
            last = tr.nodes[-1]
            node = current_nodes[idx]
            candidate_idx = gpu_candidates.get(ai, []) if gpu_candidates is not None else _query_neighbors(last, grid, cs)
            merged_idx = []

            # split-aware combine: allow one previous mask to map to multiple components (1->N) in current slice.
            if gap == 1:
                ref = _track_reference_node(tr)
                cur_node = node
                cur_iou, cur_ov_prev, cur_ov_cur = _node_overlap_pair(ref, cur_node)
                cur_area_err = abs(cur_node.area - ref.area) / (ref.area + 1e-6)
                improved = True
                while improved:
                    improved = False
                    best_j = None
                    best_candidate = None
                    best_err = cur_area_err
                    best_ov_prev = cur_ov_prev
                    split_candidates = _query_neighbors(cur_node, grid, cs)
                    split_candidates = list(dict.fromkeys(candidate_idx + split_candidates))
                    for j in split_candidates:
                        if j in used or j == idx or j in merged_idx:
                            continue
                        n2 = current_nodes[j]
                        if n2.z != cur_node.z:
                            continue
                        dist2 = np.linalg.norm(n2.centroid - cur_node.centroid)
                        if dist2 > (merge_dist * 2.4):
                            continue
                        test = _merge_nodes([cur_node, n2])
                        test_iou, test_ov_prev, _ = _node_overlap_pair(ref, test)
                        test_err = abs(test.area - ref.area) / (ref.area + 1e-6)
                        if (test_ov_prev > best_ov_prev + 0.08) or (test_err < best_err - 0.04 and test_iou >= cur_iou - 0.12):
                            best_j = j
                            best_candidate = test
                            best_err = test_err
                            best_ov_prev = test_ov_prev
                    if best_j is not None:
                        merged_idx.append(best_j)
                        cur_node = best_candidate
                        cur_iou, cur_ov_prev, cur_ov_cur = _node_overlap_pair(ref, cur_node)
                        cur_area_err = best_err
                        improved = True
                if merged_idx:
                    node = cur_node
                    iou = cur_iou

            tr.nodes.append(node)
            tr.links.append(iou)
            skips = max(0, gap - 1)
            tr.gap_bridges += skips
            if skips > 0:
                tr.gap_hist[skips] = tr.gap_hist.get(skips, 0) + 1
            used.add(idx)
            for j in merged_idx:
                used.add(j)
            claimed_tracks.add(ai)
            assigned_track_ai[ai] = idx

            node_debug[idx].update({
                "track_id": int(tr.track_id),
                "status": "assigned",
                "reason": "best_scored_edge",
                "score": float(score),
            })
            for j in merged_idx:
                node_debug[j].update({
                    "track_id": int(tr.track_id),
                    "status": "merged",
                    "reason": f"split_component_merged_into_{idx}",
                    "score": float(score),
                })

        # Force-attach leftover nodes: prioritize nearest track continuity and do not leave leftovers.
        leftover = [i for i, n in enumerate(current_nodes) if i not in used]
        for idx in sorted(leftover, key=lambda j: current_nodes[j].area, reverse=True):
            node = current_nodes[idx]
            best = None
            best_score = -1e9
            for ai, tr in enumerate(active):
                last = tr.nodes[-1]
                gap = node.z - last.z
                # if track already linked at this z, allow merge-in rather than append
                if gap == 0 and ai in assigned_track_ai:
                    dist = np.linalg.norm(node.centroid - last.centroid)
                    if dist > (merge_dist * 2.5):
                        continue
                    score = -0.004 * dist
                    if score > best_score:
                        best_score = score
                        best = (ai, tr, 0)
                    continue

                if ai in claimed_tracks:
                    continue
                if gap < 1 or gap > max_gap:
                    continue
                dist = np.linalg.norm(node.centroid - last.centroid)
                if dist > (link_dist * 2.8):
                    continue
                ref = _track_reference_node(tr)
                area_ratio = min(ref.area, node.area) / max(ref.area, node.area)
                _, ov_prev, ov_cur = _node_overlap_pair(ref, node)
                score = 0.25 * area_ratio + 0.35 * ov_prev + 0.15 * ov_cur - 0.0038 * dist - 0.012 * (gap - 1)
                if score > best_score:
                    best_score = score
                    best = (ai, tr, gap)
            if best is None and len(active):
                # absolute fallback: attach to nearest active track to avoid leftovers.
                dmin, ai_min = None, None
                for ai2, tr2 in enumerate(active):
                    d = np.linalg.norm(node.centroid - tr2.nodes[-1].centroid)
                    if dmin is None or d < dmin:
                        dmin, ai_min = d, ai2
                if ai_min is not None:
                    best = (ai_min, active[ai_min], max(1, node.z - active[ai_min].nodes[-1].z))
                    best_score = -0.003 * float(dmin)

            if best is not None:
                ai, tr, gap = best
                last = tr.nodes[-1]
                if gap == 0:
                    merged = _merge_nodes([last, node])
                    tr.nodes[-1] = merged
                    if len(tr.nodes) > 1 and len(tr.links) > 0:
                        tr.links[-1] = _node_iou(tr.nodes[-2], merged)
                else:
                    iou = _node_iou(last, node)
                    tr.nodes.append(node)
                    tr.links.append(iou)
                    skips = max(0, gap - 1)
                    tr.gap_bridges += skips
                    if skips > 0:
                        tr.gap_hist[skips] = tr.gap_hist.get(skips, 0) + 1
                    claimed_tracks.add(ai)
                    assigned_track_ai[ai] = idx
                used.add(idx)
                node_debug[idx].update({
                    "track_id": int(tr.track_id),
                    "status": "forced_attach",
                    "reason": "leftover_attached_to_nearest_track",
                    "score": float(best_score),
                })

        for i, node in enumerate(current_nodes):
            if i not in used:
                if anchor_first_slice:
                    reason = "omitted_small_unmatched" if node.area < force_attach_min_area else "omitted_unmatched_to_first_slice"
                    node_debug[i].update({
                        "track_id": None,
                        "status": "omitted",
                        "reason": reason,
                        "score": best_score_by_node.get(i, None),
                    })
                    continue

                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)

                if i in candidate_nodes:
                    reason = "omitted_by_higher_score_or_track_claim"
                else:
                    reason = "no_valid_link_started_new_track"
                node_debug[i].update({
                    "track_id": int(tr.track_id),
                    "status": "new_track",
                    "reason": reason,
                    "score": best_score_by_node.get(i, None),
                })

        if return_debug:
            track_debug = []
            for ai, tr in enumerate(active):
                last = tr.nodes[-1]
                if z - last.z > max_gap:
                    status = "inactive_gap"
                elif ai in claimed_tracks:
                    status = "linked"
                else:
                    cand = gpu_candidates.get(ai, []) if gpu_candidates is not None else _query_neighbors(last, grid, cs)
                    status = "no_link" if len(cand) else "no_neighbor_candidate"
                track_debug.append({
                    "track_id": int(tr.track_id),
                    "last_z": int(last.z),
                    "status": status,
                })
            debug_records.append({"z": int(z), "nodes": node_debug, "tracks": track_debug})

        # memory reuse: periodically compact historical masks on long active tracks.
        if z % 8 == 0:
            _compact_track_history(active, keep_recent=2)

        active = [t for t in active if z - t.nodes[-1].z < max_gap]

    if return_debug:
        return tracks, debug_records
    return tracks
