from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import cv2


LOGGER = logging.getLogger(__name__)


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


def _node_overlap_scores(node_a: InstanceNode, node_b: InstanceNode,
                         slice_mask_a: Optional[np.ndarray] = None,
                         slice_mask_b: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Return directed overlap scores.

    IoU A = |A∩B| / |A| where A is the previous/track mask.
    IoU B = |A∩B| / |B| where B is the current candidate mask.
    """
    ay0, ay1, ax0, ax1 = node_a.bbox
    by0, by1, bx0, bx1 = node_b.bbox
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return 0.0, 0.0

    a_crop = _node_mask_crop(node_a, slice_mask_a)
    b_crop = _node_mask_crop(node_b, slice_mask_b)
    a_view = a_crop[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
    b_view = b_crop[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    inter = np.logical_and(a_view, b_view).sum()
    iou_a = float(inter / node_a.area) if node_a.area > 0 else 0.0
    iou_b = float(inter / node_b.area) if node_b.area > 0 else 0.0
    return iou_a, iou_b


def _node_iou_b(node_anchor: InstanceNode, node_candidate: InstanceNode,
                slice_mask_anchor: Optional[np.ndarray] = None,
                slice_mask_candidate: Optional[np.ndarray] = None) -> float:
    """IoU_B from raw binary instance masks (not bbox/polygon approximation)."""
    _, iou_b = _node_overlap_scores(node_anchor, node_candidate, slice_mask_anchor, slice_mask_candidate)
    return iou_b


def _node_intersection(node_a: InstanceNode, node_b: InstanceNode,
                       slice_mask_a: Optional[np.ndarray] = None,
                       slice_mask_b: Optional[np.ndarray] = None) -> int:
    ay0, ay1, ax0, ax1 = node_a.bbox
    by0, by1, bx0, bx1 = node_b.bbox
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return 0

    a_crop = _node_mask_crop(node_a, slice_mask_a)
    b_crop = _node_mask_crop(node_b, slice_mask_b)
    a_view = a_crop[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
    b_view = b_crop[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    return int(np.logical_and(a_view, b_view).sum())


def _node_iou(node_a: InstanceNode, node_b: InstanceNode, slice_mask_a: Optional[np.ndarray] = None, slice_mask_b: Optional[np.ndarray] = None) -> float:
    # Backward-compatible undirected IoU helper for callers that still expect it.
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


def _polygon_similarity(node_a: InstanceNode, node_b: InstanceNode) -> float:
    # contour-based similarity, independent of bbox-overlap hard gate
    if not node_a.contours or not node_b.contours:
        return 0.0
    c1 = max(node_a.contours, key=cv2.contourArea)
    c2 = max(node_b.contours, key=cv2.contourArea)
    try:
        # lower is better; convert to [0,1]
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
    if ys.size:
        centroid = np.array([ys.mean() + y0, xs.mean() + x0], dtype=np.float32)
    else:
        centroid = np.array([0.0, 0.0], dtype=np.float32)
    node = InstanceNode(
        z=z,
        instance_id=int(min(ids)),
        bbox=(y0, y1, x0, x1),
        centroid=centroid,
        area=int(merged.sum()),
        mask_crop=merged,
        contours=_extract_contours(merged),
    )
    return node


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


def _recent_track_nodes(track: Track, depth: int = 3) -> List[InstanceNode]:
    if depth <= 0:
        return []
    return track.nodes[-depth:][::-1]


def _best_anchor_metrics(track: Track, node: InstanceNode, max_gap: int, depth: int = 3):
    """Pick best recent track node (up to `depth`) for matching this candidate node."""
    best = None
    for anchor in _recent_track_nodes(track, depth):
        gap = node.z - anchor.z
        if gap < 1 or gap > max_gap:
            continue
        inter = _node_intersection(anchor, node)
        iou_a, iou_b = _node_overlap_scores(anchor, node)
        dist = float(np.linalg.norm(node.centroid - anchor.centroid))
        if best is None:
            best = (anchor, gap, inter, iou_a, iou_b, dist)
            continue
        _, _, b_inter, b_iou_a, _, b_dist = best
        if (iou_a > b_iou_a) or (iou_a == b_iou_a and inter > b_inter) or (
            iou_a == b_iou_a and inter == b_inter and dist < b_dist
        ):
            best = (anchor, gap, inter, iou_a, iou_b, dist)
    return best




def _has_attachable_graph_for_node(node: InstanceNode,
                                  active_tracks: List[Track],
                                  max_gap: int,
                                  link_dist: float,
                                  exclude_track_id: Optional[int] = None) -> bool:
    """Return True if `node` can still be attached to another existing track."""
    for tr in active_tracks:
        if exclude_track_id is not None and tr.track_id == exclude_track_id:
            continue
        anchor_metrics = _best_anchor_metrics(tr, node, max_gap=max_gap, depth=3)
        if anchor_metrics is None:
            continue
        _, gap, inter, _, _, dist = anchor_metrics
        dist_gate = link_dist * (1.0 + min(1.0, 0.5 * (gap - 1)))
        radius_gate = 2.0 * float(max(1, node.area))
        if inter > 0 or dist <= max(dist_gate, radius_gate):
            return True
    return False

def _gpu_prefilter_candidates(active, current_nodes, link_dist, size_tolerance):
    try:
        import torch
        if not torch.cuda.is_available() or len(active) == 0 or len(current_nodes) == 0:
            return None
        a_cent = np.stack([t.nodes[-1].centroid for t in active]).astype(np.float32)
        c_cent = np.stack([n.centroid for n in current_nodes]).astype(np.float32)
        a_area = np.array([t.nodes[-1].area for t in active], dtype=np.float32)
        c_area = np.array([n.area for n in current_nodes], dtype=np.float32)

        A = torch.from_numpy(a_cent).to("cuda")
        C = torch.from_numpy(c_cent).to("cuda")
        dists = torch.cdist(A, C)
        aa = torch.from_numpy(a_area).to("cuda")[:, None]
        ca = torch.from_numpy(c_area).to("cuda")[None, :]
        area_ratio = torch.minimum(aa, ca) / torch.maximum(aa, ca)
        valid = (dists <= link_dist) & (area_ratio >= (size_tolerance * 0.7))
        valid_cpu = valid.detach().cpu().numpy()
        return {i: np.where(valid_cpu[i])[0].tolist() for i in range(valid_cpu.shape[0])}
    except Exception:
        return None


def _compute_pairwise_distances(active_tracks: List[Track], nodes: List[InstanceNode], use_gpu: bool):
    if len(active_tracks) == 0 or len(nodes) == 0:
        return np.zeros((len(active_tracks), len(nodes)), dtype=np.float32)

    a_cent = np.stack([t.nodes[-1].centroid for t in active_tracks]).astype(np.float32)
    c_cent = np.stack([n.centroid for n in nodes]).astype(np.float32)
    if not use_gpu:
        return np.linalg.norm(a_cent[:, None, :] - c_cent[None, :, :], axis=2).astype(np.float32)

    try:
        import torch

        if not torch.cuda.is_available():
            return np.linalg.norm(a_cent[:, None, :] - c_cent[None, :, :], axis=2).astype(np.float32)
        A = torch.from_numpy(a_cent).to("cuda")
        C = torch.from_numpy(c_cent).to("cuda")
        return torch.cdist(A, C).detach().cpu().numpy().astype(np.float32)
    except Exception:
        return np.linalg.norm(a_cent[:, None, :] - c_cent[None, :, :], axis=2).astype(np.float32)


def build_association_tracks(slice_masks, link_iou=0.1, link_dist=30.0, size_tolerance=0.6,
                             max_gap=3, gpu_prefilter=False, merge_dist=12.0, merge_iou_b_min=0.2):
    tracks: List[Track] = []
    active: List[Track] = []
    next_track_id = 1

    for z in range(len(slice_masks)):
        current_nodes = _extract_nodes_for_slice(slice_masks[z], z)
        used = set()

        # Hard guarantee: NEVER drop any first-slice masks.
        if z == 0:
            for node in current_nodes:
                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)
            continue

        grid, cs = _build_spatial_grid(current_nodes, link_dist)
        gpu_candidates = _gpu_prefilter_candidates(active, current_nodes, link_dist, size_tolerance) if gpu_prefilter else None
        assigned = []

        # 1) Primary linking: ONLY previous slice anchors (z-1), globally rank pairings by IoU_A then distance.
        # This avoids per-track greedy assignment and improves matching consistency.
        prev_tracks = [tr for tr in active if tr.nodes and tr.nodes[-1].z == (z - 1)]
        active_pos = {id(tr): i for i, tr in enumerate(active)}
        pairwise_dist = _compute_pairwise_distances(prev_tracks, current_nodes, use_gpu=gpu_prefilter)
        pair_candidates = []
        for ti, tr in enumerate(prev_tracks):
            last = tr.nodes[-1]
            candidate_idx = gpu_candidates.get(active_pos[id(tr)], []) if gpu_candidates is not None else _query_neighbors(last, grid, cs)
            widened = [ci for ci in range(len(current_nodes)) if pairwise_dist[ti, ci] <= (link_dist * 2.0)]
            for ci in set(candidate_idx).union(widened):
                dist = float(pairwise_dist[ti, ci])
                if dist > (link_dist * 2.0):
                    continue
                node = current_nodes[ci]
                iou_a, _ = _node_overlap_scores(last, node)
                pair_candidates.append((iou_a, -dist, ti, ci, last))

        pair_candidates.sort(reverse=True)
        used_prev_tracks = set()
        for iou_a, neg_dist, ti, ci, anchor in pair_candidates:
            if ti in used_prev_tracks or ci in used:
                continue
            tr = prev_tracks[ti]
            node = current_nodes[ci]
            tr.nodes.append(node)
            tr.links.append(iou_a)
            assigned.append((tr, node, anchor, 1))
            used.add(ci)
            used_prev_tracks.add(ti)

        # 2) Leftovers attach using two-slices-before anchors (z-2) for tracks absent at z-1.
        for i, node in enumerate(current_nodes):
            if i in used:
                continue
            best = None
            for tr in active:
                last = tr.nodes[-1]
                if last.z != (z - 2):
                    continue
                dist = float(np.linalg.norm(node.centroid - last.centroid))
                if dist > (link_dist * 2.0):
                    continue
                iou_a, _ = _node_overlap_scores(last, node)
                if best is None or (iou_a > best[2]) or (iou_a == best[2] and dist < best[3]):
                    best = (tr, last, iou_a, dist)
            if best is not None:
                tr, anchor, iou_a, _ = best
                tr.nodes.append(node)
                tr.links.append(iou_a)
                assigned.append((tr, node, anchor, 2))
                used.add(i)

        filled_track_ids = {tr.track_id for tr, _, _, _ in assigned}
        hanging_track_ids = {
            tr.track_id for tr in active if tr.nodes and tr.nodes[-1].z in (z - 1, z - 2)
        } - filled_track_ids

        # 3) Merge fallback: only after all eligible z-1/z-2 graphs are filled.
        if hanging_track_ids:
            LOGGER.info(
                "semi3d merge bypassed at z=%d because hanging graphs remain: %s",
                z,
                sorted(hanging_track_ids),
            )
        else:
            for tr, base_node, anchor, gap in assigned:
                if gap not in (1, 2):
                    continue
                best_merge_j = None
                best_merge_iou_b = float(merge_iou_b_min)
                for j, n2 in enumerate(current_nodes):
                    if j in used:
                        continue
                    merge_dist_gate = min(float(merge_dist), 1.5 * link_dist)
                    if np.linalg.norm(n2.centroid - base_node.centroid) > merge_dist_gate:
                        continue
                    # keep attach-first: if this node can still attach to any graph, do not merge it away.
                    if _has_attachable_graph_for_node(n2, active, max_gap=max_gap, link_dist=link_dist, exclude_track_id=tr.track_id):
                        continue
                    iou_b_prev = _node_iou_b(anchor, n2)
                    iou_b_two = 0.0
                    if tr.nodes and len(tr.nodes) >= 2:
                        for old in tr.nodes[:-1][::-1]:
                            if old.z == (z - 2):
                                iou_b_two = _node_iou_b(old, n2)
                                break
                    iou_b = max(iou_b_prev, iou_b_two)
                    if iou_b > best_merge_iou_b:
                        best_merge_iou_b = iou_b
                        best_merge_j = j
                if best_merge_j is not None:
                    merged = _merge_nodes([base_node, current_nodes[best_merge_j]])
                    tr.nodes[-1] = merged
                    iou_a, _ = _node_overlap_scores(anchor, merged)
                    tr.links[-1] = iou_a
                    used.add(best_merge_j)

        # 4) If still unmatched, try any remaining graph via nearest/overlap before new track.
        leftovers = [i for i in range(len(current_nodes)) if i not in used]
        for i in leftovers:
            node = current_nodes[i]
            best_tr, best_anchor, best_gap, best_iou, best_dist = None, None, None, -1.0, None
            for tr in active:
                anchor_metrics = _best_anchor_metrics(tr, node, max_gap=max_gap, depth=3)
                if anchor_metrics is None:
                    continue
                anchor, gap, inter, iou_a, _, dist = anchor_metrics
                if inter <= 0 and dist > 2.0 * float(max(1, node.area)):
                    continue
                if (iou_a > best_iou) or (iou_a == best_iou and (best_dist is None or dist < best_dist)):
                    best_tr, best_anchor, best_gap, best_iou, best_dist = tr, anchor, gap, iou_a, dist
            if best_tr is None:
                continue
            best_tr.nodes.append(node)
            best_tr.links.append(best_iou)
            skips = max(0, best_gap - 1)
            best_tr.gap_bridges += skips
            if skips > 0:
                best_tr.gap_hist[skips] = best_tr.gap_hist.get(skips, 0) + 1
            used.add(i)

        # Apply skip bookkeeping for phase1/phase2 assignments.
        for tr, _, _, gap in assigned:
            skips = max(0, gap - 1)
            tr.gap_bridges += skips
            if skips > 0:
                tr.gap_hist[skips] = tr.gap_hist.get(skips, 0) + 1

        for i, node in enumerate(current_nodes):
            if i not in used:
                if active:
                    diagnostics = []
                    for tr in active:
                        anchor_metrics = _best_anchor_metrics(tr, node, max_gap=max_gap, depth=3)
                        if anchor_metrics is None:
                            continue
                        anchor, gap, inter, iou_a, iou_b, dist = anchor_metrics
                        diagnostics.append((dist, tr.track_id, anchor.z, gap, inter, iou_a, iou_b))
                    diagnostics.sort(key=lambda x: x[0])
                    top = diagnostics[:3]
                    LOGGER.warning(
                        "semi3d creating new track z=%d inst=%d after failed connection checks; top_candidates=%s",
                        z,
                        node.instance_id,
                        [
                            {
                                "track_id": tid,
                                "anchor_z": az,
                                "gap": gap,
                                "dist": round(dist, 4),
                                "inter": inter,
                                "iou_a": round(iou_a, 6),
                                "iou_b": round(iou_b, 6),
                            }
                            for dist, tid, az, gap, inter, iou_a, iou_b in top
                        ],
                    )
                else:
                    LOGGER.warning(
                        "semi3d creating new track z=%d inst=%d because there are zero active tracks",
                        z,
                        node.instance_id,
                    )
                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)

        active = [t for t in active if z - t.nodes[-1].z < max_gap]

    return tracks
