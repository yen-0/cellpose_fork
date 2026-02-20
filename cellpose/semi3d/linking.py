from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class InstanceNode:
    z: int
    instance_id: int
    mask: np.ndarray
    centroid: np.ndarray
    area: int


@dataclass
class Track:
    track_id: int
    nodes: List[InstanceNode] = field(default_factory=list)
    links: List[float] = field(default_factory=list)
    gap_bridges: int = 0


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def extract_nodes(slice_masks: List[np.ndarray]) -> List[InstanceNode]:
    nodes: List[InstanceNode] = []
    for z, masks in enumerate(slice_masks):
        ids = np.unique(masks)
        ids = ids[ids > 0]
        for inst_id in ids:
            m = masks == inst_id
            coords = np.argwhere(m)
            centroid = coords.mean(axis=0) if coords.size else np.array([0.0, 0.0])
            nodes.append(InstanceNode(z=z, instance_id=int(inst_id), mask=m, centroid=centroid, area=int(m.sum())))
    return nodes


def build_association_tracks(
    slice_masks: List[np.ndarray],
    link_iou: float = 0.1,
    link_dist: float = 30.0,
    size_tolerance: float = 0.6,
    max_gap: int = 2,
):
    nodes = extract_nodes(slice_masks)
    nodes_by_z: Dict[int, List[InstanceNode]] = {}
    for n in nodes:
        nodes_by_z.setdefault(n.z, []).append(n)

    tracks: List[Track] = []
    active: List[Track] = []
    next_track_id = 1

    for z in range(len(slice_masks)):
        current_nodes = nodes_by_z.get(z, [])
        used = set()
        for tr in active:
            last = tr.nodes[-1]
            best = None
            best_score = -1.0
            for node in current_nodes:
                if id(node) in used:
                    continue
                gap = node.z - last.z
                if gap < 1 or gap > max_gap:
                    continue
                dist = np.linalg.norm(node.centroid - last.centroid)
                if dist > link_dist:
                    continue
                iou = mask_iou(last.mask, node.mask)
                area_ratio = min(last.area, node.area) / max(last.area, node.area)
                if iou < link_iou and gap == 1:
                    continue
                if area_ratio < size_tolerance:
                    continue
                score = iou + 0.5 * area_ratio - 0.01 * dist - 0.05 * (gap - 1)
                if score > best_score:
                    best_score = score
                    best = (node, iou, gap)
            if best is not None:
                node, iou, gap = best
                tr.nodes.append(node)
                tr.links.append(iou)
                tr.gap_bridges += max(0, gap - 1)
                used.add(id(node))

        for node in current_nodes:
            if id(node) not in used:
                tr = Track(track_id=next_track_id, nodes=[node])
                next_track_id += 1
                tracks.append(tr)
                active.append(tr)

        # keep active tracks that might still receive links
        active = [t for t in active if z - t.nodes[-1].z < max_gap]

    return tracks
