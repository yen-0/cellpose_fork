import argparse
import os
import random
import numpy as np
from cellpose import io, models
from .linking import build_association_tracks
from .refine import relabel_tracks
from .training.train import run_training
from .evaluation.eval import run_evaluation


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


def build_parser():
    p = argparse.ArgumentParser("cellpose semi3d tools")
    sub = p.add_subparsers(dest="command", required=True)

    inf = sub.add_parser("semi3d")
    inf.add_argument("--input", required=True)
    inf.add_argument("--output", required=True)
    inf.add_argument("--pretrained_model", default="cpsam")
    inf.add_argument("--use_gpu", action="store_true")
    inf.add_argument("--diameter", type=float, default=None)
    inf.add_argument("--seed", type=int, default=0)
    inf.add_argument("--link-iou", dest="link_iou", type=float, default=0.1)
    inf.add_argument("--link-dist", dest="link_dist", type=float, default=30.0)
    inf.add_argument("--size-tolerance", dest="size_tolerance", type=float, default=0.6)
    inf.add_argument("--max-gap", dest="max_gap", type=int, default=2)
    inf.add_argument("--min-track-len", dest="min_track_len", type=int, default=2)
    inf.add_argument("--min-conf", dest="min_conf", type=float, default=0.05)
    inf.add_argument("--save-3d-labels", action="store_true")

    tr = sub.add_parser("semi3d-train")
    tr.add_argument("--input", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--pretrained_model", default="cpsam")
    tr.add_argument("--use_gpu", action="store_true")
    tr.add_argument("--diameter", type=float, default=None)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--simulate-z-dropout", action="store_true")
    tr.add_argument("--dropout-prob", type=float, default=0.2)
    tr.add_argument("--learning-rate", type=float, default=1e-2)
    tr.add_argument("--epochs", type=int, default=200)

    ev = sub.add_parser("semi3d-eval")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--gt", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--seed", type=int, default=0)

    return p


def run_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "semi3d":
        total, kept = _run_inference(args)
        print(f"semi3d complete: tracks={total}, kept={kept}")
    elif args.command == "semi3d-train":
        path, n = run_training(args)
        print(f"semi3d training complete: samples={n}, model={path}")
    elif args.command == "semi3d-eval":
        path, metrics = run_evaluation(args)
        print(f"semi3d eval complete: {path}")
        print(metrics)
