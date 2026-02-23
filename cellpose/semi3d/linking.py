from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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


def build_association_tracks(slice_masks, link_iou=0.1, link_dist=30.0, size_tolerance=0.6,
                             max_gap=3, gpu_prefilter=False, merge_dist=12.0):
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

        for ai, tr in enumerate(active):
            last = tr.nodes[-1]
            if z - last.z > max_gap:
                continue
            candidate_idx = gpu_candidates.get(ai, []) if gpu_candidates is not None else _query_neighbors(last, grid, cs)

            best = None
            best_score = -1e9
            for i in candidate_idx:
                if i in used:
                    continue
                node = current_nodes[i]
                gap = node.z - last.z
                if gap < 1 or gap > max_gap:
                    continue
                dist = np.linalg.norm(node.centroid - last.centroid)
                if dist > (link_dist * (1.0 + min(1.0, 0.5 * (gap - 1)))):
                    continue
                area_ratio = min(last.area, node.area) / max(last.area, node.area)
                poly_sim = _polygon_similarity(last, node)
                iou = _node_iou(last, node)
                # Never hard-reject by low IoU: keep weak-overlap motion matches eligible.
                score = iou + 0.30 * area_ratio - 0.006 * dist - 0.015 * (gap - 1) + 0.05 * poly_sim
                if score > best_score:
                    best_score = score
                    best = (i, node, iou, gap)

            if best is not None:
                idx, node, iou, gap = best
                merge_nodes = [node]
                if gap == 1:
                    for j in candidate_idx:
                        if j in used or j == idx:
                            continue
                        n2 = current_nodes[j]
                        if np.linalg.norm(n2.centroid - node.centroid) <= merge_dist:
                            test = _merge_nodes([node, n2])
                            merged_iou = _node_iou(last, test)
                            # Merge decision based ONLY on IoU improvement.
                            if merged_iou > iou:
                                merge_nodes.append(n2)
                                used.add(j)
                                node = test
                                iou = merged_iou
                    if len(merge_nodes) > 1:
                        node = _merge_nodes(merge_nodes)
                        iou = _node_iou(last, node)

                tr.nodes.append(node)
                tr.links.append(iou)
                skips = max(0, gap - 1)
                tr.gap_bridges += skips
                if skips > 0:
                    tr.gap_hist[skips] = tr.gap_hist.get(skips, 0) + 1
                used.add(idx)

        # second-pass forced linking for remaining nodes: attach to nearest viable active track
        leftovers = [i for i in range(len(current_nodes)) if i not in used]
        for i in leftovers:
            node = current_nodes[i]
            best_ai, best_gap, best_dist = None, None, None
            for ai, tr in enumerate(active):
                last = tr.nodes[-1]
                gap = node.z - last.z
                if gap < 1 or gap > max_gap:
                    continue
                dist = np.linalg.norm(node.centroid - last.centroid)
                if dist > (2.0 * link_dist):
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_ai = ai
                    best_gap = gap
            if best_ai is None:
                continue
            tr = active[best_ai]
            last = tr.nodes[-1]
            iou = _node_iou(last, node)
            tr.nodes.append(node)
            tr.links.append(iou)
            skips = max(0, best_gap - 1)
            tr.gap_bridges += skips
            if skips > 0:
                tr.gap_hist[skips] = tr.gap_hist.get(skips, 0) + 1
            used.add(i)

        for i, node in enumerate(current_nodes):
            if i not in used:
                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)

        active = [t for t in active if z - t.nodes[-1].z < max_gap]

    return tracks
