from dataclasses import dataclass, field
import copy
from typing import Dict, List, Optional, Tuple
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


LOGGER = logging.getLogger(__name__)


_GPU_PREFILTER_CUDA_UNAVAILABLE_WARNED = False
_GPU_PREFILTER_EXCEPTION_WARNED = False
_GPU_PREFILTER_ACTIVE_LOGGED = False

_LINK_WORKERS_AUTO_LOGGED = False
_LINK_WORKERS_MULTICORE_LOGGED = False
_LINK_WORKERS_SINGLECORE_LOGGED = False


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




def _node_raw_iou(node_a: InstanceNode, node_b: InstanceNode,
                  slice_mask_a: Optional[np.ndarray] = None,
                  slice_mask_b: Optional[np.ndarray] = None) -> float:
    """Raw binary-mask IoU on true instance pixels (not bbox IoU)."""
    inter = _node_raw_intersection(node_a, node_b, slice_mask_a, slice_mask_b)
    union = int(node_a.area) + int(node_b.area) - int(inter)
    return float(inter / union) if union > 0 else 0.0


def _distance_gate_from_area(area: int, link_dist: float, scale: float = 2.0) -> float:
    """Area-aware distance gate that grows sub-linearly with object size."""
    return max(float(link_dist), scale * np.sqrt(float(max(1, area))))


def _node_raw_intersection(node_a: InstanceNode, node_b: InstanceNode,
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


def _node_raw_iou_b(node_anchor: InstanceNode, node_candidate: InstanceNode,
                    slice_mask_anchor: Optional[np.ndarray] = None,
                    slice_mask_candidate: Optional[np.ndarray] = None) -> float:
    inter = _node_raw_intersection(node_anchor, node_candidate, slice_mask_anchor, slice_mask_candidate)
    return float(inter / node_candidate.area) if node_candidate.area > 0 else 0.0


def _recompute_track_links(track: Track):
    links = []
    if len(track.nodes) >= 2:
        for prev, cur in zip(track.nodes[:-1], track.nodes[1:]):
            iou_a, raw_iou_b = _node_overlap_scores(prev, cur)
            raw_iou = _node_raw_iou(prev, cur)
            links.append(max(iou_a, raw_iou_b, raw_iou))
    track.links = links
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


def _extract_nodes_for_slice(slice_mask: np.ndarray, z: int, exclude_edge_touching: bool = False) -> List[InstanceNode]:
    nodes: List[InstanceNode] = []
    ids = np.unique(slice_mask)
    ids = ids[ids > 0]
    for inst_id in ids:
        ys, xs = np.where(slice_mask == inst_id)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if exclude_edge_touching:
            h, w = slice_mask.shape[:2]
            touches_edge = (y0 == 0) or (x0 == 0) or (y1 >= h) or (x1 >= w)
            if touches_edge:
                continue
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
                                  min_iou_a: float,
                                  exclude_track_id: Optional[int] = None) -> bool:
    """Return True if `node` can still be attached to another existing track."""
    for tr in active_tracks:
        if exclude_track_id is not None and tr.track_id == exclude_track_id:
            continue
        anchor_metrics = _best_anchor_metrics(tr, node, max_gap=max_gap, depth=3)
        if anchor_metrics is None:
            continue
        anchor, gap, inter, iou_a, _, dist = anchor_metrics
        if iou_a < float(min_iou_a):
            continue
        raw_iou = _node_raw_iou(anchor, node)
        if raw_iou < (0.5 * float(min_iou_a)):
            continue
        dist_gate = link_dist * (1.0 + min(1.0, 0.5 * (gap - 1)))
        area_gate = _distance_gate_from_area(node.area, link_dist, scale=2.0)
        if inter > 0 or dist <= min(1.5 * dist_gate, area_gate):
            return True
    return False

def _gpu_prefilter_candidates(active, current_nodes, link_dist, size_tolerance):
    global _GPU_PREFILTER_CUDA_UNAVAILABLE_WARNED, _GPU_PREFILTER_EXCEPTION_WARNED
    try:
        import torch
        if len(active) == 0 or len(current_nodes) == 0:
            return None
        if not torch.cuda.is_available():
            if not _GPU_PREFILTER_CUDA_UNAVAILABLE_WARNED:
                LOGGER.warning(
                    "[semi3d:stage2] --semi3d_link_gpu_prefilter requested but CUDA unavailable; "
                    "falling back to CPU candidate scoring"
                )
                _GPU_PREFILTER_CUDA_UNAVAILABLE_WARNED = True
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
        global _GPU_PREFILTER_ACTIVE_LOGGED
        if not _GPU_PREFILTER_ACTIVE_LOGGED:
            LOGGER.info(
                "[semi3d:stage2] GPU candidate prefilter active (CUDA available): "
                "tracks=%d, nodes=%d",
                len(active),
                len(current_nodes),
            )
            _GPU_PREFILTER_ACTIVE_LOGGED = True
        return {i: np.where(valid_cpu[i])[0].tolist() for i in range(valid_cpu.shape[0])}
    except Exception as e:
        if not _GPU_PREFILTER_EXCEPTION_WARNED:
            LOGGER.warning(
                "[semi3d:stage2] --semi3d_link_gpu_prefilter fallback to CPU due to GPU prefilter error: %s",
                e,
            )
            _GPU_PREFILTER_EXCEPTION_WARNED = True
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


def _pair_candidates_for_track(track_idx: int, anchor: InstanceNode, current_nodes: List[InstanceNode],
                               link_iou: float):
    candidates = []
    if len(current_nodes) == 0:
        return candidates

    ay0, ay1, ax0, ax1 = anchor.bbox
    bb = np.asarray([n.bbox for n in current_nodes], dtype=np.int32)
    by0 = bb[:, 0]
    by1 = bb[:, 1]
    bx0 = bb[:, 2]
    bx1 = bb[:, 3]

    iy0 = np.maximum(ay0, by0)
    iy1 = np.minimum(ay1, by1)
    ix0 = np.maximum(ax0, bx0)
    ix1 = np.minimum(ax1, bx1)
    ih = np.maximum(0, iy1 - iy0)
    iw = np.maximum(0, ix1 - ix0)
    inter_bbox = (ih * iw).astype(np.float32)

    if anchor.area > 0:
        iou_a_upper = inter_bbox / float(anchor.area)
    else:
        iou_a_upper = np.zeros_like(inter_bbox)
    areas = np.asarray([n.area for n in current_nodes], dtype=np.float32)
    union_lower = np.maximum(1.0, float(anchor.area) + areas - inter_bbox)
    raw_iou_upper = inter_bbox / union_lower
    viable = np.where(np.maximum(iou_a_upper, raw_iou_upper) >= float(link_iou))[0]

    for ci in viable.tolist():
        node = current_nodes[ci]
        iou_a, _ = _node_overlap_scores(anchor, node)
        raw_iou = _node_raw_iou(anchor, node)
        score = max(iou_a, raw_iou)
        if score < float(link_iou):
            continue
        candidates.append((score, raw_iou, iou_a, track_idx, ci, anchor))
    return candidates


def build_association_tracks(slice_masks, link_iou=0.1, link_dist=30.0, size_tolerance=0.6,
                             max_gap=3, gpu_prefilter=False, merge_dist=12.0, merge_iou_b_min=0.35,
                             merge_competition_margin=0.1, short_track_merge_len=2,
                             short_track_merge_iou_b_min=None,
                             show_progress=False, link_workers=1,
                             exclude_edge_touching=False,
                             allow_new_tracks_after_first_slice=True,
                             return_debug_steps=False):
    tracks: List[Track] = []
    active: List[Track] = []
    next_track_id = 1

    debug_steps = None
    if return_debug_steps:
        debug_steps = {
            "step1_iou_adjacent": np.zeros_like(slice_masks, dtype=np.int32),
            "step2_iou_z2": np.zeros_like(slice_masks, dtype=np.int32),
            "step3_iou_fallback": np.zeros_like(slice_masks, dtype=np.int32),
        }

    z_iter = tqdm(range(len(slice_masks)), desc="[semi3d:stage2] linking slices", unit="slice") if show_progress else range(len(slice_masks))
    for z in z_iter:
        current_nodes = _extract_nodes_for_slice(slice_masks[z], z, exclude_edge_touching=exclude_edge_touching)
        used = set()

        if z == 0:
            for node in current_nodes:
                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)
            continue

        assigned = []

        # 1-3) Node-centric linking: iterate every current mask, score against existing graphs
        # with IoU + distance, and use exponential decay over past slices to keep track state.
        worker_count = int(link_workers) if link_workers is not None else 1
        global _LINK_WORKERS_AUTO_LOGGED, _LINK_WORKERS_MULTICORE_LOGGED, _LINK_WORKERS_SINGLECORE_LOGGED
        if worker_count <= 0:
            worker_count = os.cpu_count() or 1
            if not _LINK_WORKERS_AUTO_LOGGED:
                LOGGER.info(
                    "[semi3d:stage2] --semi3d_link_workers=%s resolved to auto multicore workers=%d",
                    link_workers,
                    worker_count,
                )
                _LINK_WORKERS_AUTO_LOGGED = True
        if worker_count > 1 and not _LINK_WORKERS_MULTICORE_LOGGED:
            LOGGER.info("[semi3d:stage2] Multicore link scoring enabled with workers=%d", worker_count)
            _LINK_WORKERS_MULTICORE_LOGGED = True
        elif worker_count <= 1 and not _LINK_WORKERS_SINGLECORE_LOGGED:
            LOGGER.info("[semi3d:stage2] Link scoring running single-core (workers=%d)", worker_count)
            _LINK_WORKERS_SINGLECORE_LOGGED = True

        recent_active = [tr for tr in active if tr.nodes and (z - tr.nodes[-1].z) <= max_gap]
        taken_track_ids = set()

        prefilter_map = None
        node_to_track_indices = None
        if gpu_prefilter:
            prefilter_map = _gpu_prefilter_candidates(recent_active, current_nodes, link_dist, size_tolerance)
            if prefilter_map is not None:
                node_to_track_indices = {i: [] for i in range(len(current_nodes))}
                for ti, node_ids in prefilter_map.items():
                    for ni in node_ids:
                        if ni in node_to_track_indices:
                            node_to_track_indices[ni].append(ti)

        def _candidate_track_indices_for_node(node_idx: int):
            if node_to_track_indices is None:
                return list(range(len(recent_active)))
            return node_to_track_indices.get(node_idx, [])

        def _decayed_match_for_node_index(node_idx: int):
            node = current_nodes[node_idx]
            best = None
            candidate_tracks = _candidate_track_indices_for_node(node_idx)
            for ti in candidate_tracks:
                tr = recent_active[ti]
                weighted_iou = 0.0
                weighted_dist = 0.0
                weight_sum = 0.0
                best_anchor = None
                best_gap = None
                best_iou = -1.0

                for anchor in _recent_track_nodes(tr, depth=4):
                    gap = node.z - anchor.z
                    if gap < 1 or gap > max_gap:
                        continue
                    iou_a, _ = _node_overlap_scores(anchor, node)
                    raw_iou = _node_raw_iou(anchor, node)
                    iou_score = max(iou_a, raw_iou)

                    dist = float(np.linalg.norm(node.centroid - anchor.centroid))
                    dist_gate = link_dist * (1.0 + min(1.0, 0.5 * (gap - 1)))
                    area_gate = _distance_gate_from_area(node.area, link_dist, scale=2.0)
                    allowed_dist = max(1.0, min(1.5 * dist_gate, area_gate))
                    dist_score = max(0.0, 1.0 - (dist / allowed_dist))

                    w = float(0.65 ** (gap - 1))
                    weighted_iou += w * iou_score
                    weighted_dist += w * dist_score
                    weight_sum += w

                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_anchor = anchor
                        best_gap = gap

                if weight_sum <= 0.0 or best_anchor is None:
                    continue

                avg_iou = weighted_iou / weight_sum
                avg_dist = weighted_dist / weight_sum
                combined = 0.8 * avg_iou + 0.2 * avg_dist

                min_combined = max(0.05, 0.7 * float(link_iou))
                if (best_iou < float(link_iou)) and (combined < min_combined):
                    continue

                if best is None or combined > best[0]:
                    best = (combined, tr, best_anchor, best_gap, best_iou)
            return node_idx, best

        order = sorted(range(len(current_nodes)), key=lambda i: current_nodes[i].area, reverse=True)
        if worker_count > 1 and len(order) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                scored = list(ex.map(_decayed_match_for_node_index, order))
        else:
            scored = [_decayed_match_for_node_index(i) for i in order]

        scored.sort(key=lambda t: (-1.0 if t[1] is None else -float(t[1][0]), t[0]))
        for i, best in scored:
            if i in used or best is None:
                continue
            _, tr, anchor, gap, best_iou = best
            if tr.track_id in taken_track_ids:
                continue

            node = current_nodes[i]
            tr.nodes.append(node)
            tr.links.append(float(best_iou))
            assigned.append((tr, node, anchor, int(gap)))
            used.add(i)
            taken_track_ids.add(tr.track_id)

            if debug_steps is not None:
                nmask = node.full_mask(slice_masks[z].shape, slice_mask=slice_masks[z])
                debug_steps["step1_iou_adjacent"][z][nmask] = tr.track_id
        filled_track_ids = {tr.track_id for tr, _, _, _ in assigned}
        filled_track_ids.update(
            tr.track_id for tr in active if tr.nodes and tr.nodes[-1].z == z
        )
        hanging_track_ids = {
            tr.track_id for tr in active if tr.nodes and tr.nodes[-1].z in (z - 1, z - 2)
        } - filled_track_ids

        def _run_merge_fallback(allow_anchor_overlap_only: bool = False):
            """Attempt to merge unused same-slice fragments into already-assigned tracks."""
            recent_active = [
                tr for tr in active if tr.nodes and tr.nodes[-1].z in (z - 1, z - 2)
            ]
            while True:
                merge_candidates = []

                def _merge_candidates_for_assignment(entry):
                    tr, base_node, anchor, gap = entry
                    if gap not in (1, 2):
                        return []
                    local = []
                    for j, n2 in enumerate(current_nodes):
                        if j in used:
                            continue
                        merge_dist_gate = min(float(merge_dist), 0.75 * link_dist)
                        local_neighbors = sum(
                            np.linalg.norm(other.nodes[-1].centroid - n2.centroid) <= (merge_dist_gate * 1.5)
                            for other in recent_active
                            if other.track_id != tr.track_id
                        )
                        merge_dist_gate /= (1.0 + 0.25 * min(4, local_neighbors))
                        if np.linalg.norm(n2.centroid - base_node.centroid) > merge_dist_gate:
                            continue
                        if _has_attachable_graph_for_node(
                            n2,
                            active,
                            max_gap=max_gap,
                            link_dist=link_dist,
                            min_iou_a=link_iou,
                            exclude_track_id=tr.track_id,
                        ):
                            continue

                        iou_b_prev = _node_iou_b(anchor, n2)
                        raw_iou_prev = _node_raw_iou(anchor, n2)
                        inter_prev = _node_raw_intersection(anchor, n2)
                        iou_b_two, raw_iou_two, inter_two = 0.0, 0.0, 0
                        if tr.nodes and len(tr.nodes) >= 2:
                            for old in tr.nodes[:-1][::-1]:
                                if old.z == (z - 2):
                                    iou_b_two = _node_iou_b(old, n2)
                                    raw_iou_two = _node_raw_iou(old, n2)
                                    inter_two = _node_raw_intersection(old, n2)
                                    break
                        iou_b = max(iou_b_prev, iou_b_two)
                        raw_iou = max(raw_iou_prev, raw_iou_two)
                        inter_raw = max(inter_prev, inter_two)
                        if allow_anchor_overlap_only and inter_raw <= 0:
                            continue

                        alt_best_iou_a = 0.0
                        for other in recent_active:
                            if other.track_id == tr.track_id:
                                continue
                            alt_metrics = _best_anchor_metrics(other, n2, max_gap=max_gap, depth=3)
                            if alt_metrics is None:
                                continue
                            alt_anchor, _, _, alt_iou_a, _, _ = alt_metrics
                            alt_best_iou_a = max(alt_best_iou_a, max(alt_iou_a, _node_raw_iou(alt_anchor, n2)))

                        adaptive_merge_iou_b_min = float(merge_iou_b_min) + (0.05 * min(4, local_neighbors))
                        required_iou = max(adaptive_merge_iou_b_min, alt_best_iou_a + float(merge_competition_margin))
                        required_raw_iou = max(0.05, 0.6 * required_iou)
                        if iou_b < required_iou or raw_iou < required_raw_iou:
                            continue
                        merge_score = min(iou_b, raw_iou)
                        local.append((merge_score, tr, anchor, base_node, j))
                    return local

                if worker_count > 1 and len(assigned) > 1 and len(current_nodes) > 0:
                    with ThreadPoolExecutor(max_workers=worker_count) as ex:
                        for local in ex.map(_merge_candidates_for_assignment, assigned):
                            merge_candidates.extend(local)
                else:
                    for entry in assigned:
                        merge_candidates.extend(_merge_candidates_for_assignment(entry))

                if not merge_candidates:
                    break

                merge_candidates.sort(reverse=True, key=lambda x: x[0])
                consumed_nodes = set()
                merged_any = False
                for _, tr, anchor, base_node, node_idx in merge_candidates:
                    if node_idx in used or node_idx in consumed_nodes:
                        continue
                    merged = _merge_nodes([base_node, current_nodes[node_idx]])
                    tr.nodes[-1] = merged
                    iou_a, _ = _node_overlap_scores(anchor, merged)
                    raw_iou = _node_raw_iou(anchor, merged)
                    tr.links[-1] = max(iou_a, raw_iou)
                    used.add(node_idx)
                    consumed_nodes.add(node_idx)
                    merged_any = True
                if not merged_any:
                    break

        # 4) Merge fallback as a separate iterative loop after linking is done.
        if hanging_track_ids:
            LOGGER.info(
                "semi3d merge bypassed at z=%d because hanging graphs remain: %s",
                z,
                sorted(hanging_track_ids),
            )
            _run_merge_fallback(allow_anchor_overlap_only=True)
        else:
            _run_merge_fallback(allow_anchor_overlap_only=False)

        for tr, _, _, gap in assigned:
            skips = max(0, gap - 1)
            tr.gap_bridges += skips
            if skips > 0:
                tr.gap_hist[skips] = tr.gap_hist.get(skips, 0) + 1

        for i, node in enumerate(current_nodes):
            if i not in used:
                if z > 0 and not allow_new_tracks_after_first_slice:
                    LOGGER.info(
                        "semi3d dropping unmatched node at z=%d inst=%d because spawning new graphs is disabled",
                        z,
                        node.instance_id,
                    )
                    continue
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
                                "gap": gp,
                                "intersection": inter,
                                "iou_a": float(ia),
                                "iou_b": float(ib),
                                "dist": float(d),
                            }
                            for d, tid, az, gp, inter, ia, ib in top
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

    tracks_before_short_merge = copy.deepcopy(tracks) if return_debug_steps else None

    # 5) Post-link short-track merge pass: merge newly-created short tracks into larger tracks
    # using raw IoU_B (raw pixel intersections) on overlapping z slices.
    if short_track_merge_iou_b_min is None:
        short_track_merge_iou_b_min = max(0.12, 0.6 * float(merge_iou_b_min))

    removed_track_ids = set()
    short_tracks = sorted(
        [tr for tr in tracks if 0 < len(tr.nodes) <= int(short_track_merge_len)],
        key=lambda t: (len(t.nodes), t.track_id),
    )

    for tr in short_tracks:
        if tr.track_id in removed_track_ids:
            continue
        best_target = None
        best_score = float(short_track_merge_iou_b_min)
        best_matches = None
        for target in tracks:
            if target.track_id == tr.track_id or target.track_id in removed_track_ids:
                continue
            if len(target.nodes) <= len(tr.nodes):
                continue
            t_by_z = {n.z: (idx, n) for idx, n in enumerate(target.nodes)}
            matched = []
            for n in tr.nodes:
                zn = t_by_z.get(n.z)
                if zn is None:
                    continue
                _, tn = zn
                iou_b_raw = _node_raw_iou_b(tn, n)
                inter_raw = _node_raw_intersection(tn, n)
                if inter_raw <= 0:
                    continue
                if iou_b_raw < float(short_track_merge_iou_b_min):
                    continue
                matched.append((n.z, iou_b_raw, inter_raw))
            if not matched:
                continue
            score = float(np.mean([m[1] for m in matched])) + 0.01 * len(matched)
            if score > best_score:
                best_score = score
                best_target = target
                best_matches = {zv for zv, _, _ in matched}

        if best_target is None or not best_matches:
            continue

        merged_any = False
        target_by_z = {n.z: idx for idx, n in enumerate(best_target.nodes)}
        for n in tr.nodes:
            if n.z not in best_matches:
                continue
            idx = target_by_z.get(n.z)
            if idx is None:
                continue
            best_target.nodes[idx] = _merge_nodes([best_target.nodes[idx], n])
            merged_any = True

        if merged_any:
            best_target.nodes.sort(key=lambda n: n.z)
            _recompute_track_links(best_target)
            removed_track_ids.add(tr.track_id)

    if removed_track_ids:
        tracks = [tr for tr in tracks if tr.track_id not in removed_track_ids]

    if return_debug_steps:
        debug_steps["step4_post_merge_fallback_tracks"] = tracks_before_short_merge
        debug_steps["step5_post_short_track_merge_tracks"] = copy.deepcopy(tracks)
        return tracks, debug_steps

    return tracks
