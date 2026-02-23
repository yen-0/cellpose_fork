import os
import random
import logging
import json
from types import SimpleNamespace
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from cellpose import io, models
from .linking import build_association_tracks
from .refine import relabel_tracks
from .training.train import run_training, extract_track_features
from .evaluation.eval import run_evaluation
from .training.model import RefinerModel

LOGGER = logging.getLogger(__name__)


def _load_stack(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg"))]
        if not files:
            raise ValueError(f"No slice images found in directory: {path}")
        return np.stack([io.imread_2D(f) for f in files], axis=0)
    return io.imread_3D(path)


def _remove_border_instances(mask, border_px=0):
    if border_px <= 0:
        return mask
    out = mask.copy()
    h, w = out.shape
    border = np.zeros_like(out, dtype=bool)
    b = min(border_px, h // 2, w // 2)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    edge_ids = np.unique(out[border])
    edge_ids = edge_ids[edge_ids > 0]
    for inst_id in edge_ids:
        out[out == inst_id] = 0
    return out




def _adapt_feature_dim(feats, refiner):
    target_dim = None
    if hasattr(refiner, "w"):
        target_dim = int(refiner.w.shape[0])
    elif hasattr(refiner, "w1"):
        target_dim = int(refiner.w1.shape[0])
    if target_dim is None or feats.shape[1] == target_dim:
        return feats
    if feats.shape[1] > target_dim:
        return feats[:, :target_dim]
    pad = np.zeros((feats.shape[0], target_dim - feats.shape[1]), dtype=feats.dtype)
    return np.concatenate([feats, pad], axis=1)


def _extract_debug_maps(flows):
    flow_slice = flows[1] if isinstance(flows, (list, tuple)) and len(flows) > 1 else None
    prob_slice = flows[2] if isinstance(flows, (list, tuple)) and len(flows) > 2 else None
    flow_mag = None
    if flow_slice is not None:
        arr = np.asarray(flow_slice)
        if arr.ndim >= 3:
            if arr.shape[0] >= 2:
                fy, fx = arr[0], arr[1]
            elif arr.shape[-1] >= 2:
                fy, fx = arr[..., 0], arr[..., 1]
            else:
                fy = fx = None
            if fy is not None:
                flow_mag = np.sqrt(fy.astype(np.float32) ** 2 + fx.astype(np.float32) ** 2)
    if prob_slice is not None:
        p = np.asarray(prob_slice).astype(np.float32)
        if p.ndim > 2:
            p = np.squeeze(p)
        prob_slice = p
    return flow_slice, flow_mag, prob_slice



def _defog_probability(prob_slice, bg_percentile=2.0, hi_percentile=98.0, gamma=0.85,
                       bg_sigma=50.0, boundary_sigma=1.2, boundary_strength=0.35):
    p = np.asarray(prob_slice, dtype=np.float32)
    if p.ndim > 2:
        p = np.squeeze(p)

    # Remove low-frequency fog field.
    if bg_sigma is not None and bg_sigma > 0:
        b = gaussian_filter(p, sigma=float(bg_sigma))
        p_flat = p - b
    else:
        p_flat = p.copy()
    p_flat = p_flat - float(np.min(p_flat))

    # Darken boundaries using gradient magnitude so borders become clearer.
    if boundary_strength > 0:
        gx = np.gradient(p_flat, axis=1)
        gy = np.gradient(p_flat, axis=0)
        gmag = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
        if boundary_sigma is not None and boundary_sigma > 0:
            gmag = gaussian_filter(gmag, sigma=float(boundary_sigma))
        gmag = gmag / (float(gmag.max()) + 1e-6)
        p_flat = p_flat - float(boundary_strength) * gmag * (float(np.percentile(p_flat, 95)) + 1e-6)
        p_flat = p_flat - float(np.min(p_flat))

    hi = float(np.percentile(p_flat, hi_percentile))
    bg = float(np.percentile(p_flat, bg_percentile))
    if hi <= bg + 1e-6:
        return np.clip(p_flat, 0.0, 1.0).astype(np.float32)

    q = (p_flat - bg) / (hi - bg + 1e-6)
    q = np.clip(q, 0.0, 1.0)
    if gamma != 1.0:
        q = np.power(q, float(gamma), dtype=np.float32)
    return q.astype(np.float32)


def _run_stage1(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    stack = _load_stack(args.input)
    os.makedirs(args.output, exist_ok=True)

    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.pretrained_model)
    per_slice_masks = []
    per_slice_flows = []
    per_slice_flow_mag = []
    per_slice_prob = []

    LOGGER.info("[semi3d:stage1] starting per-slice inference (%d slices)", stack.shape[0])
    for z in range(stack.shape[0]):
        _, flows, *_ = model.eval(stack[z], do_3D=False, diameter=args.diameter, cellprob_threshold=getattr(args, "stage1_cellprob_threshold", 0.0))
        flow_slice, flow_mag, prob_slice = _extract_debug_maps(flows)
        per_slice_flows.append(flow_slice)
        per_slice_flow_mag.append(flow_mag)

        if prob_slice is not None and getattr(args, "stage1_defog_prob", True):
            prob_slice = _defog_probability(
                prob_slice,
                bg_percentile=getattr(args, "stage1_prob_bg_percentile", 2.0),
                hi_percentile=getattr(args, "stage1_prob_hi_percentile", 98.0),
                gamma=getattr(args, "stage1_prob_gamma", 0.85),
                bg_sigma=getattr(args, "stage1_prob_bg_sigma", 50.0),
                boundary_sigma=getattr(args, "stage1_prob_boundary_sigma", 1.2),
                boundary_strength=getattr(args, "stage1_prob_boundary_strength", 0.35),
            )

        if prob_slice is not None and getattr(args, "stage1_use_defog_prob_for_cellpose", True):
            masks, *_ = model.eval(prob_slice.astype(np.float32), do_3D=False, diameter=args.diameter,
                                   cellprob_threshold=getattr(args, "stage1_cellprob_threshold", 0.0))
        else:
            masks, *_ = model.eval(stack[z], do_3D=False, diameter=args.diameter,
                                   cellprob_threshold=getattr(args, "stage1_cellprob_threshold", 0.0))

        masks = _remove_border_instances(masks.astype(np.int32), args.border_exclusion_px)
        per_slice_masks.append(masks)
        per_slice_prob.append(prob_slice)
        LOGGER.info("[semi3d:stage1] slice %d/%d", z + 1, stack.shape[0])

    masks_path = os.path.join(args.output, "semi3d_stage1_masks.tif")
    io.imsave(masks_path, np.stack(per_slice_masks, axis=0).astype(np.int32))
    if args.save_flows:
        np.save(os.path.join(args.output, "semi3d_stage1_flows.npy"), np.array(per_slice_flows, dtype=object), allow_pickle=True)
    if any(p is not None for p in per_slice_prob):
        prob_stack = np.stack([np.zeros_like(per_slice_masks[0], dtype=np.float32) if p is None else p.astype(np.float32) for p in per_slice_prob], axis=0)
        io.imsave(os.path.join(args.output, "semi3d_stage1_prob.tif"), prob_stack)
    if getattr(args, "save_debug_tiff", False):
        if any(f is not None for f in per_slice_flow_mag):
            flow_mag_stack = np.stack([np.zeros_like(per_slice_masks[0], dtype=np.float32) if f is None else f.astype(np.float32) for f in per_slice_flow_mag], axis=0)
            io.imsave(os.path.join(args.output, "semi3d_stage1_flow_mag.tif"), flow_mag_stack)
    LOGGER.info("[semi3d:stage1] complete -> %s", masks_path)
    return masks_path


def _load_stage1_masks(path, memmap=False):
    lower = path.lower()
    if lower.endswith((".tif", ".tiff")):
        return io.imread(path)
    if lower.endswith(".npy"):
        return np.load(path, mmap_mode="r" if memmap else None)
    raise ValueError(f"Unsupported stage1 mask format: {path}")


def _predict_keep_prob(refiner, feats, use_gpu=False):
    if not use_gpu:
        return refiner.predict_proba(feats)
    try:
        import torch
        if not torch.cuda.is_available():
            LOGGER.warning("[semi3d:stage2] GPU requested for stage2 refiner scoring but CUDA unavailable; falling back to CPU")
            return refiner.predict_proba(feats)

        x = torch.from_numpy(feats.astype(np.float32)).to("cuda")
        if hasattr(refiner, "w") and hasattr(refiner, "b"):
            w = torch.from_numpy(np.asarray(refiner.w, dtype=np.float32)).to("cuda")
            b = torch.tensor(float(refiner.b), device="cuda", dtype=torch.float32)
            z = x @ w + b
            p = torch.sigmoid(z)
            return p.detach().cpu().numpy()
        if hasattr(refiner, "w1") and hasattr(refiner, "b1") and hasattr(refiner, "w2") and hasattr(refiner, "b2"):
            w1 = torch.from_numpy(np.asarray(refiner.w1, dtype=np.float32)).to("cuda")
            b1 = torch.from_numpy(np.asarray(refiner.b1, dtype=np.float32)).to("cuda")
            w2 = torch.from_numpy(np.asarray(refiner.w2, dtype=np.float32)).to("cuda")
            b2 = torch.tensor(float(refiner.b2), device="cuda", dtype=torch.float32)
            h = torch.tanh(x @ w1 + b1)
            z = h @ w2 + b2
            p = torch.sigmoid(z)
            return p.detach().cpu().numpy()
    except Exception as e:
        LOGGER.warning("[semi3d:stage2] GPU scoring fallback to CPU due to: %s", e)
    return refiner.predict_proba(feats)


def _save_link_debug_overlay(per_slice_masks, debug_records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    code_map = {
        "best_scored_edge": 1,
        "merged": 2,
        "new_track": 3,
        "omitted_by_higher_score_or_track_claim": 4,
        "no_valid_link_started_new_track": 5,
    }
    overlays = []
    reason_stack = []
    id_stack = []

    for z, sm in enumerate(per_slice_masks):
        h, w = sm.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[sm > 0] = np.array([40, 40, 40], dtype=np.uint8)
        reason_map = np.zeros((h, w), dtype=np.uint8)
        tid_map = np.zeros((h, w), dtype=np.int32)

        rec = debug_records[z] if z < len(debug_records) else {"nodes": []}
        for node in rec.get("nodes", []):
            inst_id = int(node.get("instance_id", 0))
            if inst_id <= 0:
                continue
            mask = (sm == inst_id)
            if not np.any(mask):
                continue
            status = str(node.get("status", ""))
            reason = str(node.get("reason", ""))
            tid = int(node.get("track_id") or 0)
            if status == "assigned":
                color = np.array([0, 220, 0], dtype=np.uint8)
                reason_val = code_map["best_scored_edge"]
            elif status == "merged":
                color = np.array([255, 180, 0], dtype=np.uint8)
                reason_val = code_map["merged"]
            elif status == "new_track":
                if reason.startswith("omitted_by_higher_score"):
                    color = np.array([0, 0, 255], dtype=np.uint8)
                    reason_val = code_map["omitted_by_higher_score_or_track_claim"]
                else:
                    color = np.array([200, 0, 200], dtype=np.uint8)
                    reason_val = code_map["new_track"]
            else:
                color = np.array([100, 100, 255], dtype=np.uint8)
                reason_val = code_map["no_valid_link_started_new_track"]

            rgb[mask] = color
            reason_map[mask] = reason_val
            if tid > 0:
                tid_map[mask] = tid

            ys, xs = np.where(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            short = reason[:18]
            text = f"i{inst_id}->t{tid}:{short}"
            cv2.putText(rgb, text, (max(0, cx - 20), max(12, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

        overlays.append(rgb)
        reason_stack.append(reason_map)
        id_stack.append(tid_map)

    io.imsave(os.path.join(output_dir, "semi3d_link_debug_overlay.tif"), np.stack(overlays, axis=0))
    io.imsave(os.path.join(output_dir, "semi3d_link_debug_reason_codes.tif"), np.stack(reason_stack, axis=0))
    io.imsave(os.path.join(output_dir, "semi3d_link_debug_track_ids.tif"), np.stack(id_stack, axis=0).astype(np.int32))
    with open(os.path.join(output_dir, "semi3d_link_debug_records.json"), "w", encoding="utf-8") as f:
        json.dump(debug_records, f, indent=2)


def _run_stage2(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    stack = _load_stack(args.input)
    os.makedirs(args.output, exist_ok=True)

    per_slice_masks = _load_stage1_masks(args.stage1_masks, memmap=args.memmap_stage2_inputs)
    per_slice_flows = None
    if args.stage1_flows and os.path.exists(args.stage1_flows):
        # object arrays cannot be memory-mapped safely; load only when explicitly provided.
        per_slice_flows = np.load(args.stage1_flows, allow_pickle=True)
    stage1_prob = None
    prob_path = args.stage1_prob if getattr(args, "stage1_prob", None) else os.path.join(args.output, "semi3d_stage1_prob.tif")
    if os.path.exists(prob_path):
        stage1_prob = io.imread(prob_path).astype(np.float32)

    LOGGER.info("[semi3d:stage2] linking tracks")
    link_debug = None
    link_out = build_association_tracks(
        per_slice_masks,
        link_iou=args.link_iou,
        link_dist=args.link_dist,
        size_tolerance=args.size_tolerance,
        max_gap=args.max_gap,
        gpu_prefilter=args.link_gpu_prefilter,
        merge_dist=args.merge_dist,
        link_workers=args.link_workers,
        return_debug=getattr(args, "save_link_debug", False),
    )
    if getattr(args, "save_link_debug", False):
        tracks, link_debug = link_out
    else:
        tracks = link_out

    if args.refiner_model is not None:
        if not os.path.exists(args.refiner_model):
            raise FileNotFoundError(f"semi3d refiner model not found: {args.refiner_model}")
        refiner = RefinerModel.load(args.refiner_model)
        if len(tracks):
            LOGGER.info("[semi3d:stage2] scoring %d tracks with %s", len(tracks), "GPU" if args.stage2_use_gpu else "CPU")
            feats = np.stack([extract_track_features(tr, stack, source_masks=per_slice_masks, all_tracks=tracks) for tr in tracks], axis=0)
            feats = _adapt_feature_dim(feats, refiner)
            keep_prob = _predict_keep_prob(refiner, feats, use_gpu=args.stage2_use_gpu)
            if getattr(args, "save_debug_tiff", False):
                prob_map = np.zeros((len(tracks), 1, 1), dtype=np.float32)
                prob_map[:, 0, 0] = keep_prob.astype(np.float32)
                io.imsave(os.path.join(args.output, "semi3d_refiner_keep_prob.tif"), prob_map)
            tracks = [tr for tr, p in zip(tracks, keep_prob) if p >= args.refiner_threshold]

    LOGGER.info("[semi3d:stage2] relabel/reconstruct")
    refined, labels3d, reconstructed_flags, kept = relabel_tracks(
        tracks,
        stack,
        flow_stack=per_slice_flows,
        min_track_len=args.min_track_len,
        min_conf=args.min_conf,
        fill_edges=args.fill_edges,
        return_track_labels=args.save_3d_labels,
        source_masks=per_slice_masks,
        avoid_occupied=not args.allow_overlap_recon,
        min_free_fraction=args.recon_min_free_fraction,
        prob_stack=stage1_prob if args.use_prob_occupancy else None,
        prob_occupancy_thresh=args.prob_occupancy_thresh,
    )

    refined_stack = np.stack(refined, axis=0).astype(np.int32)
    io.imsave(os.path.join(args.output, "semi3d_refined_masks.tif"), refined_stack)
    if args.save_3d_labels and labels3d is not None:
        io.imsave(os.path.join(args.output, "semi3d_track_labels.tif"), labels3d.astype(np.int32))
    io.imsave(os.path.join(args.output, "semi3d_reconstructed_flags.tif"), reconstructed_flags.astype(np.uint8))
    if getattr(args, "save_link_debug", False) and link_debug is not None:
        _save_link_debug_overlay(per_slice_masks, link_debug, args.output)
    LOGGER.info("[semi3d:stage2] complete")
    return len(tracks), len(kept)


def _run_inference(args):
    masks_path = _run_stage1(args)
    stage2_args = SimpleNamespace(**vars(args))
    stage2_args.stage1_masks = masks_path
    stage2_args.stage1_flows = os.path.join(args.output, "semi3d_stage1_flows.npy") if args.save_flows else None
    stage2_args.stage1_prob = os.path.join(args.output, "semi3d_stage1_prob.tif")
    return _run_stage2(stage2_args)


def run_from_cellpose_args(args):
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    mode_flags = [args.semi3d, args.semi3d_train, args.semi3d_eval, args.semi3d_stage1, args.semi3d_stage2]
    if sum(bool(v) for v in mode_flags) != 1:
        raise ValueError("Choose exactly one semi3d mode")

    if args.semi3d:
        if not args.semi3d_input or not args.semi3d_output:
            raise ValueError("--semi3d requires --semi3d_input and --semi3d_output")
        semi_args = SimpleNamespace(
            input=args.semi3d_input,
            output=args.semi3d_output,
            pretrained_model=args.semi3d_pretrained_model,
            use_gpu=args.use_gpu,
            diameter=args.semi3d_diameter,
            seed=args.semi3d_seed,
            link_iou=args.semi3d_link_iou,
            link_dist=args.semi3d_link_dist,
            size_tolerance=args.semi3d_size_tolerance,
            max_gap=args.semi3d_max_gap,
            min_track_len=args.semi3d_min_track_len,
            min_conf=args.semi3d_min_conf,
            save_3d_labels=args.semi3d_save_3d_labels,
            refiner_model=args.semi3d_refiner_model,
            refiner_threshold=args.semi3d_refiner_threshold,
            border_exclusion_px=args.semi3d_border_exclusion_px,
            fill_edges=args.semi3d_fill_edges,
            save_flows=args.semi3d_save_flows,
            stage2_use_gpu=args.semi3d_stage2_use_gpu,
            memmap_stage2_inputs=args.semi3d_memmap_stage2_inputs,
            link_gpu_prefilter=args.semi3d_link_gpu_prefilter,
            merge_dist=args.semi3d_merge_dist,
            link_workers=args.semi3d_link_workers,
            allow_overlap_recon=args.semi3d_allow_overlap_recon,
            recon_min_free_fraction=args.semi3d_recon_min_free_fraction,
            save_debug_tiff=args.semi3d_save_debug_tiff,
            save_link_debug=args.semi3d_save_link_debug,
            stage1_prob=args.semi3d_stage1_prob,
            use_prob_occupancy=args.semi3d_use_prob_occupancy,
            prob_occupancy_thresh=args.semi3d_prob_occupancy_thresh,
            stage1_defog_prob=not args.semi3d_disable_stage1_prob_defog,
            stage1_prob_bg_percentile=args.semi3d_stage1_prob_bg_percentile,
            stage1_prob_hi_percentile=args.semi3d_stage1_prob_hi_percentile,
            stage1_prob_gamma=args.semi3d_stage1_prob_gamma,
            stage1_prob_bg_sigma=args.semi3d_stage1_prob_bg_sigma,
            stage1_cellprob_threshold=args.semi3d_stage1_cellprob_threshold,
            stage1_use_defog_prob_for_cellpose=not args.semi3d_disable_stage1_use_defog_prob_for_cellpose,
            stage1_prob_boundary_sigma=args.semi3d_stage1_prob_boundary_sigma,
            stage1_prob_boundary_strength=args.semi3d_stage1_prob_boundary_strength,
        )
        total, kept = _run_inference(semi_args)
        print(f"semi3d complete: tracks={total}, kept={kept}")

    elif args.semi3d_stage1:
        if not args.semi3d_input or not args.semi3d_output:
            raise ValueError("--semi3d_stage1 requires --semi3d_input and --semi3d_output")
        semi_args = SimpleNamespace(
            input=args.semi3d_input,
            output=args.semi3d_output,
            pretrained_model=args.semi3d_pretrained_model,
            use_gpu=args.use_gpu,
            diameter=args.semi3d_diameter,
            seed=args.semi3d_seed,
            border_exclusion_px=args.semi3d_border_exclusion_px,
            save_flows=args.semi3d_save_flows,
            save_debug_tiff=args.semi3d_save_debug_tiff,
            save_link_debug=args.semi3d_save_link_debug,
            stage1_prob=args.semi3d_stage1_prob,
            use_prob_occupancy=args.semi3d_use_prob_occupancy,
            prob_occupancy_thresh=args.semi3d_prob_occupancy_thresh,
            stage1_defog_prob=not args.semi3d_disable_stage1_prob_defog,
            stage1_prob_bg_percentile=args.semi3d_stage1_prob_bg_percentile,
            stage1_prob_hi_percentile=args.semi3d_stage1_prob_hi_percentile,
            stage1_prob_gamma=args.semi3d_stage1_prob_gamma,
            stage1_prob_bg_sigma=args.semi3d_stage1_prob_bg_sigma,
            stage1_cellprob_threshold=args.semi3d_stage1_cellprob_threshold,
            stage1_use_defog_prob_for_cellpose=not args.semi3d_disable_stage1_use_defog_prob_for_cellpose,
            stage1_prob_boundary_sigma=args.semi3d_stage1_prob_boundary_sigma,
            stage1_prob_boundary_strength=args.semi3d_stage1_prob_boundary_strength,
        )
        path = _run_stage1(semi_args)
        print(f"semi3d stage1 complete: {path}")

    elif args.semi3d_stage2:
        if not args.semi3d_input or not args.semi3d_output or not args.semi3d_stage1_masks:
            raise ValueError("--semi3d_stage2 requires --semi3d_input, --semi3d_output and --semi3d_stage1_masks")
        semi_args = SimpleNamespace(
            input=args.semi3d_input,
            output=args.semi3d_output,
            seed=args.semi3d_seed,
            stage1_masks=args.semi3d_stage1_masks,
            stage1_flows=args.semi3d_stage1_flows,
            link_iou=args.semi3d_link_iou,
            link_dist=args.semi3d_link_dist,
            size_tolerance=args.semi3d_size_tolerance,
            max_gap=args.semi3d_max_gap,
            min_track_len=args.semi3d_min_track_len,
            min_conf=args.semi3d_min_conf,
            save_3d_labels=args.semi3d_save_3d_labels,
            refiner_model=args.semi3d_refiner_model,
            refiner_threshold=args.semi3d_refiner_threshold,
            fill_edges=args.semi3d_fill_edges,
            stage2_use_gpu=args.semi3d_stage2_use_gpu,
            memmap_stage2_inputs=args.semi3d_memmap_stage2_inputs,
            link_gpu_prefilter=args.semi3d_link_gpu_prefilter,
            merge_dist=args.semi3d_merge_dist,
            link_workers=args.semi3d_link_workers,
            allow_overlap_recon=args.semi3d_allow_overlap_recon,
            recon_min_free_fraction=args.semi3d_recon_min_free_fraction,
            save_debug_tiff=args.semi3d_save_debug_tiff,
            save_link_debug=args.semi3d_save_link_debug,
            stage1_prob=args.semi3d_stage1_prob,
            use_prob_occupancy=args.semi3d_use_prob_occupancy,
            prob_occupancy_thresh=args.semi3d_prob_occupancy_thresh,
        )
        total, kept = _run_stage2(semi_args)
        print(f"semi3d stage2 complete: tracks={total}, kept={kept}")

    elif args.semi3d_train:
        if not args.semi3d_input or not args.semi3d_output:
            raise ValueError("--semi3d_train requires --semi3d_input and --semi3d_output")
        semi_args = SimpleNamespace(
            input=args.semi3d_input,
            output=args.semi3d_output,
            pretrained_model=args.semi3d_pretrained_model,
            use_gpu=args.use_gpu,
            diameter=args.semi3d_diameter,
            seed=args.semi3d_seed,
            simulate_z_dropout=args.semi3d_simulate_z_dropout,
            dropout_prob=args.semi3d_dropout_prob,
            learning_rate=args.semi3d_learning_rate,
            epochs=args.semi3d_epochs,
            refiner_nonlinear=not args.semi3d_refiner_linear,
            refiner_hidden_dim=args.semi3d_refiner_hidden_dim,
            synthetic_per_obj=args.semi3d_synthetic_per_obj,
            batch_size=args.semi3d_refiner_batch_size,
            verbose=args.verbose,
        )
        path, n = run_training(semi_args)
        print(f"semi3d training complete: samples={n}, model={path}")

    else:
        if not args.semi3d_pred or not args.semi3d_gt or not args.semi3d_output:
            raise ValueError("--semi3d_eval requires --semi3d_pred, --semi3d_gt and --semi3d_output")
        semi_args = SimpleNamespace(
            pred=args.semi3d_pred,
            gt=args.semi3d_gt,
            output=args.semi3d_output,
            seed=args.semi3d_seed,
        )
        path, metrics = run_evaluation(semi_args)
        print(f"semi3d eval complete: {path}")
        print(metrics)
