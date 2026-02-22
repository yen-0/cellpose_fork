import os
import json
import numpy as np
from cellpose import models
from .dataset import load_stacks
from .augment import simulate_z_dropout
from .model import LinearRefiner, NonLinearRefiner
from ..linking import build_association_tracks
from ..confidence import track_confidence


def _node_mask(node, shape, source_masks=None):
    if hasattr(node, "mask"):
        return node.mask
    src = None if source_masks is None else source_masks[node.z]
    if hasattr(node, "full_mask"):
        return node.full_mask(shape, slice_mask=src)
    raise AttributeError("node does not provide mask/full_mask")


def _track_conflict_features(track, tracks, image_shape, source_masks=None):
    overlaps = []
    same_z_conflicts = 0
    checked = 0
    for n in track.nodes:
        m = _node_mask(n, image_shape, source_masks=source_masks)
        checked += 1
        has_conflict = False
        for ot in tracks:
            if ot is track:
                continue
            for on in ot.nodes:
                if on.z != n.z:
                    continue
                om = _node_mask(on, image_shape, source_masks=source_masks)
                inter = np.logical_and(m, om).sum()
                if inter > 0:
                    union = np.logical_or(m, om).sum()
                    overlaps.append(float(inter / (union + 1e-6)))
                    has_conflict = True
                    break
            if has_conflict:
                break
        if has_conflict:
            same_z_conflicts += 1

    overlap_mean = float(np.mean(overlaps)) if overlaps else 0.0
    conflict_ratio = float(same_z_conflicts / (checked + 1e-6))
    return overlap_mean, conflict_ratio


def extract_track_features(track, image_stack, source_masks=None, all_tracks=None):
    overlap_mean, conflict_ratio = (0.0, 0.0)
    if all_tracks is not None:
        overlap_mean, conflict_ratio = _track_conflict_features(
            track, all_tracks, image_stack[0].shape[:2], source_masks=source_masks
        )
    return np.array([
        len(track.nodes),
        np.mean(track.links) if track.links else 0.0,
        np.std([n.area for n in track.nodes]) if len(track.nodes) > 1 else 0.0,
        track.gap_bridges,
        track_confidence(track, image_stack, source_masks=source_masks),
        overlap_mean,
        conflict_ratio,
    ], dtype=np.float32)


def _extract_features_and_labels(image_stack, gt_stack, pred_stack):
    tracks = build_association_tracks(pred_stack)
    x, y = [], []
    gt_present = np.array([np.any(g > 0) for g in gt_stack], dtype=np.float32)
    for tr in tracks:
        zmin, zmax = tr.nodes[0].z, tr.nodes[-1].z
        support = float(gt_present[zmin:zmax + 1].mean())
        x.append(extract_track_features(tr, image_stack, source_masks=pred_stack, all_tracks=tracks))
        y.append(1.0 if support > 0.3 else 0.0)
    if not x:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(x), np.array(y, dtype=np.float32)


def run_training(args):
    images, gts = load_stacks(args.input)
    if len(images) == 0:
        raise ValueError("No training stacks with paired masks found")
    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.pretrained_model)

    xs, ys = [], []
    for i, (img, gt) in enumerate(zip(images, gts)):
        if args.simulate_z_dropout:
            gt = simulate_z_dropout(gt, dropout_prob=args.dropout_prob, seed=args.seed + i)
        pred = []
        for z in range(img.shape[0]):
            masks, *_ = model.eval(img[z], do_3D=False, diameter=args.diameter)
            pred.append(masks.astype(np.int32))
        x, y = _extract_features_and_labels(img, gt, pred)
        if len(x):
            xs.append(x)
            ys.append(y)

    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    if getattr(args, "refiner_nonlinear", True):
        clf = NonLinearRefiner(n_features=x_train.shape[1], hidden_dim=getattr(args, "refiner_hidden_dim", 16))
    else:
        clf = LinearRefiner(n_features=x_train.shape[1])
    losses = clf.fit(x_train, y_train, lr=args.learning_rate, epochs=args.epochs)
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "semi3d_refiner.npz")
    clf.save(out_path)

    best_epoch = int(np.argmin(losses)) + 1 if len(losses) else 0
    best_loss = float(np.min(losses)) if len(losses) else None
    log = {"loss_per_epoch": [float(v) for v in losses], "best_epoch": best_epoch, "best_loss": best_loss}
    with open(os.path.join(args.output, "semi3d_refiner_train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    if getattr(args, "verbose", False):
        for i, loss in enumerate(losses, start=1):
            print(f"[semi3d:train] epoch {i}/{len(losses)} loss={loss:.6f}")
        if best_loss is not None:
            print(f"[semi3d:train] best epoch={best_epoch} loss={best_loss:.6f}")

    return out_path, len(x_train)
