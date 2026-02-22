import os
import random
import logging
from types import SimpleNamespace
import numpy as np
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


def _run_stage1(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    stack = _load_stack(args.input)
    os.makedirs(args.output, exist_ok=True)

    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.pretrained_model)
    per_slice_masks = []
    per_slice_flows = []

    LOGGER.info("[semi3d:stage1] starting per-slice inference (%d slices)", stack.shape[0])
    for z in range(stack.shape[0]):
        masks, flows, *_ = model.eval(stack[z], do_3D=False, diameter=args.diameter)
        masks = _remove_border_instances(masks.astype(np.int32), args.border_exclusion_px)
        per_slice_masks.append(masks)
        per_slice_flows.append(flows[1] if isinstance(flows, (list, tuple)) and len(flows) > 1 else None)
        LOGGER.info("[semi3d:stage1] slice %d/%d", z + 1, stack.shape[0])

    masks_path = os.path.join(args.output, "semi3d_stage1_masks.tif")
    io.imsave(masks_path, np.stack(per_slice_masks, axis=0).astype(np.int32))
    if args.save_flows:
        np.save(os.path.join(args.output, "semi3d_stage1_flows.npy"), np.array(per_slice_flows, dtype=object), allow_pickle=True)
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

    LOGGER.info("[semi3d:stage2] linking tracks")
    tracks = build_association_tracks(
        per_slice_masks,
        link_iou=args.link_iou,
        link_dist=args.link_dist,
        size_tolerance=args.size_tolerance,
        max_gap=args.max_gap,
        gpu_prefilter=args.link_gpu_prefilter,
    )

    if args.refiner_model is not None:
        if not os.path.exists(args.refiner_model):
            raise FileNotFoundError(f"semi3d refiner model not found: {args.refiner_model}")
        refiner = RefinerModel.load(args.refiner_model)
        if len(tracks):
            LOGGER.info("[semi3d:stage2] scoring %d tracks with %s", len(tracks), "GPU" if args.stage2_use_gpu else "CPU")
            feats = np.stack([extract_track_features(tr, stack, source_masks=per_slice_masks) for tr in tracks], axis=0)
            keep_prob = _predict_keep_prob(refiner, feats, use_gpu=args.stage2_use_gpu)
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
    )

    refined_stack = np.stack(refined, axis=0).astype(np.int32)
    io.imsave(os.path.join(args.output, "semi3d_refined_masks.tif"), refined_stack)
    if args.save_3d_labels and labels3d is not None:
        io.imsave(os.path.join(args.output, "semi3d_track_labels.tif"), labels3d.astype(np.int32))
    io.imsave(os.path.join(args.output, "semi3d_reconstructed_flags.tif"), reconstructed_flags.astype(np.uint8))
    LOGGER.info("[semi3d:stage2] complete")
    return len(tracks), len(kept)


def _run_inference(args):
    masks_path = _run_stage1(args)
    stage2_args = SimpleNamespace(**vars(args))
    stage2_args.stage1_masks = masks_path
    stage2_args.stage1_flows = os.path.join(args.output, "semi3d_stage1_flows.npy") if args.save_flows else None
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
