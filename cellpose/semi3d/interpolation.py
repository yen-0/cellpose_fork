from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class RecoveryResult:
    z: int
    mask: np.ndarray
    confidence: float
    bridged_only: bool


def _to_gray(image_slice: np.ndarray) -> np.ndarray:
    raw = image_slice.astype(np.float32)
    if raw.ndim == 3:
        raw = raw.mean(axis=-1)
    return raw


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    return outside - inside


def interpolate_mask_prior(prev_mask: np.ndarray, next_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Distance-transform interpolation prior between two binary masks."""
    prev_sdf = _signed_distance(prev_mask)
    next_sdf = _signed_distance(next_mask)
    sdf_mid = (1.0 - alpha) * prev_sdf + alpha * next_sdf
    return sdf_mid < 0


def _flow_to_vector(flow_slice: np.ndarray):
    if flow_slice is None:
        return None
    arr = np.asarray(flow_slice)
    if arr.ndim < 3:
        return None
    if arr.shape[0] >= 2:  # channels-first
        fy = arr[0].astype(np.float32)
        fx = arr[1].astype(np.float32)
    elif arr.shape[-1] >= 2:  # channels-last
        fy = arr[..., 0].astype(np.float32)
        fx = arr[..., 1].astype(np.float32)
    else:
        return None
    return fy, fx


def refine_mask_with_slice(
    prior_mask: np.ndarray,
    image_slice: np.ndarray,
    flow_slice: np.ndarray = None,
    iterations: int = 8,
    smooth_kernel: int = 5,
) -> np.ndarray:
    """Constrained active-contour-like deformation from prior toward image/flow evidence."""
    raw = _to_gray(image_slice)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    edges = cv2.Laplacian(raw, cv2.CV_32F, ksize=3)
    edges = cv2.GaussianBlur(np.abs(edges), (3, 3), 0)
    edges = edges / (edges.max() + 1e-6)

    prior_u8 = (prior_mask > 0).astype(np.uint8)
    work = prior_u8.copy()
    flow_vec = _flow_to_vector(flow_slice)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel, smooth_kernel))
    for _ in range(iterations):
        contour = cv2.morphologyEx(work, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        if np.any(contour):
            expand = contour & (edges < 0.35)
            shrink = contour & (edges > 0.65)
            work[expand] = 1
            work[shrink] = 0

        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

        # Constrain deformation to stay near the prior.
        dist_to_prior = cv2.distanceTransform((1 - prior_u8), cv2.DIST_L2, 5)
        work[dist_to_prior > 8] = 0

        # Optional weak advection from flow field.
        if flow_vec is not None:
            fy, fx = flow_vec
            ys, xs = np.where(work > 0)
            if ys.size:
                ny = np.clip((ys + 0.25 * fy[ys, xs]).astype(np.int32), 0, work.shape[0] - 1)
                nx = np.clip((xs + 0.25 * fx[ys, xs]).astype(np.int32), 0, work.shape[1] - 1)
                adv = np.zeros_like(work)
                adv[ny, nx] = 1
                work = np.maximum(work, adv)

    return work > 0


def interpolate_missing_mask(prev_node, next_node, z_target: int, image_slice: np.ndarray, flow_slice: np.ndarray = None):
    span = max(1, next_node.z - prev_node.z)
    alpha = (z_target - prev_node.z) / span
    prior = interpolate_mask_prior(prev_node.mask, next_node.mask, alpha)
    refined = refine_mask_with_slice(prior, image_slice, flow_slice=flow_slice)

    raw = _to_gray(image_slice)
    inside = raw[refined]
    outside = raw[~refined]
    if inside.size == 0 or outside.size == 0:
        evidence = 0.0
    else:
        evidence = float((inside.mean() - outside.mean()) / (raw.std() + 1e-6))
    geom_consistency = 1.0 - min(1.0, np.linalg.norm(prev_node.centroid - next_node.centroid) / 100.0)
    confidence = float(0.5 * geom_consistency + 0.5 * max(0.0, min(1.0, evidence)))
    return refined, confidence
