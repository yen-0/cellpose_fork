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
from .training.model import LinearRefiner


def _load_stack(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg"))]
        if not files:
            raise ValueError(f"No slice images found in directory: {path}")
        return np.stack([io.imread_2D(f) for f in files], axis=0)
    return io.imread_3D(path)


def _run_inference(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    stack = _load_stack(args.input)
    os.makedirs(args.output, exist_ok=True)

    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.pretrained_model)
    per_slice_masks = []
    per_slice_flows = []
    for z in range(stack.shape[0]):
        masks, flows, *_ = model.eval(stack[z], do_3D=False, diameter=args.diameter)
        per_slice_masks.append(masks.astype(np.int32))
        per_slice_flows.append(flows[1] if isinstance(flows, (list, tuple)) and len(flows) > 1 else None)

    tracks = build_association_tracks(
        per_slice_masks,
        link_iou=args.link_iou,
        link_dist=args.link_dist,
        size_tolerance=args.size_tolerance,
        max_gap=args.max_gap,
    )

    if args.refiner_model is not None:
        if not os.path.exists(args.refiner_model):
            raise FileNotFoundError(f"semi3d refiner model not found: {args.refiner_model}")
        refiner = LinearRefiner.load(args.refiner_model)
        if len(tracks):
            feats = np.stack([extract_track_features(tr, stack) for tr in tracks], axis=0)
            keep_prob = refiner.predict_proba(feats)
            tracks = [tr for tr, p in zip(tracks, keep_prob) if p >= args.refiner_threshold]

    refined, labels3d, reconstructed_flags, kept = relabel_tracks(
        tracks,
        stack,
        flow_stack=per_slice_flows,
        min_track_len=args.min_track_len,
        min_conf=args.min_conf,
    )

    refined_stack = np.stack(refined, axis=0).astype(np.int32)
    io.imsave(os.path.join(args.output, "semi3d_refined_masks.tif"), refined_stack)
    if args.save_3d_labels:
        io.imsave(os.path.join(args.output, "semi3d_track_labels.tif"), labels3d.astype(np.int32))
    io.imsave(os.path.join(args.output, "semi3d_reconstructed_flags.tif"), reconstructed_flags.astype(np.uint8))

    return len(tracks), len(kept)


def run_from_cellpose_args(args):
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    mode_flags = [args.semi3d, args.semi3d_train, args.semi3d_eval]
    if sum(bool(v) for v in mode_flags) != 1:
        raise ValueError("Choose exactly one of --semi3d, --semi3d_train, --semi3d_eval")

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
        )
        total, kept = _run_inference(semi_args)
        print(f"semi3d complete: tracks={total}, kept={kept}")

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
