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


def _extract_prob_from_flows(flows):
    if isinstance(flows, (list, tuple)) and len(flows) > 2:
        p = np.asarray(flows[2]).astype(np.float32)
        return np.squeeze(p)
    return None


def _local_patch_stats(mask, image_slice, prob_slice=None, patch_radius=16):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0, 0.0
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    y0 = max(0, cy - patch_radius)
    y1 = min(mask.shape[0], cy + patch_radius + 1)
    x0 = max(0, cx - patch_radius)
    x1 = min(mask.shape[1], cx + patch_radius + 1)
    p_mask = mask[y0:y1, x0:x1]

    im = image_slice
    if im.ndim == 3:
        im = im.mean(axis=-1)
    p_im = im[y0:y1, x0:x1].astype(np.float32)
    inside = p_im[p_mask]
    outside = p_im[~p_mask]
    contrast = float((inside.mean() - outside.mean()) / (p_im.std() + 1e-6)) if inside.size and outside.size else 0.0

    if prob_slice is None:
        return contrast, 0.0, 0.0
    p_prob = prob_slice[y0:y1, x0:x1].astype(np.float32)
    prob_in = float(p_prob[p_mask].mean()) if np.any(p_mask) else 0.0
    ring = np.logical_and(~p_mask, (p_prob > 0))
    prob_ring = float(p_prob[ring].mean()) if np.any(ring) else 0.0
    return contrast, prob_in, prob_ring


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


def extract_track_features(track, image_stack, source_masks=None, all_tracks=None, prob_stack=None, patch_radius=16):
    overlap_mean, conflict_ratio = (0.0, 0.0)
    if all_tracks is not None:
        overlap_mean, conflict_ratio = _track_conflict_features(track, all_tracks, image_stack[0].shape[:2], source_masks=source_masks)

    contrasts, probs_in, probs_ring = [], [], []
    for n in track.nodes:
        m = _node_mask(n, image_stack[0].shape[:2], source_masks=source_masks)
        pz = None if prob_stack is None else prob_stack[n.z]
        c, pin, pr = _local_patch_stats(m, image_stack[n.z], prob_slice=pz, patch_radius=patch_radius)
        contrasts.append(c)
        probs_in.append(pin)
        probs_ring.append(pr)

    return np.array([
        len(track.nodes),
        np.mean(track.links) if track.links else 0.0,
        np.std([n.area for n in track.nodes]) if len(track.nodes) > 1 else 0.0,
        track.gap_bridges,
        track_confidence(track, image_stack, source_masks=source_masks),
        overlap_mean,
        conflict_ratio,
        float(np.mean(contrasts)) if contrasts else 0.0,
        float(np.mean(probs_in)) if probs_in else 0.0,
        float(np.mean(probs_ring)) if probs_ring else 0.0,
    ], dtype=np.float32)


def _shift_mask(mask, dy, dx):
    out = np.zeros_like(mask)
    y0 = max(0, dy); y1 = mask.shape[0] + min(0, dy)
    x0 = max(0, dx); x1 = mask.shape[1] + min(0, dx)
    sy0 = max(0, -dy); sy1 = sy0 + (y1 - y0)
    sx0 = max(0, -dx); sx1 = sx0 + (x1 - x0)
    out[y0:y1, x0:x1] = mask[sy0:sy1, sx0:sx1]
    return out


def _synthetic_refiner_samples(image_stack, gt_stack, prob_stack=None, seed=0, n_per_obj=6):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for z in range(gt_stack.shape[0]):
        gt = gt_stack[z]
        ids = np.unique(gt)
        ids = ids[ids > 0]
        for inst_id in ids:
            target = (gt == inst_id)
            others = (gt > 0) & (~target)
            for _ in range(n_per_obj):
                dy = int(rng.integers(-6, 7)); dx = int(rng.integers(-6, 7))
                prop = _shift_mask(target, dy, dx)
                inter_t = np.logical_and(prop, target).sum()
                union_t = np.logical_or(prop, target).sum()
                iou_t = float(inter_t / (union_t + 1e-6))
                overlap_others = float(np.logical_and(prop, others).sum() / (prop.sum() + 1e-6))
                conflict_ratio = 1.0 if np.logical_and(prop, others).any() else 0.0
                c, pin, pr = _local_patch_stats(prop, image_stack[z], prob_slice=None if prob_stack is None else prob_stack[z], patch_radius=16)
                feat = np.array([1.0, iou_t, 0.0, 0.0, max(0.0, iou_t), overlap_others, conflict_ratio, c, pin, pr], dtype=np.float32)
                label = 1.0 if (iou_t > 0.6 and overlap_others < 0.05) else 0.0
                xs.append(feat); ys.append(label)
    if not xs:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(xs), np.array(ys, dtype=np.float32)


def _extract_features_and_labels(image_stack, gt_stack, pred_stack, prob_stack=None):
    tracks = build_association_tracks(pred_stack)
    x, y = [], []
    gt_present = np.array([np.any(g > 0) for g in gt_stack], dtype=np.float32)
    for tr in tracks:
        zmin, zmax = tr.nodes[0].z, tr.nodes[-1].z
        support = float(gt_present[zmin:zmax + 1].mean())
        x.append(extract_track_features(tr, image_stack, source_masks=pred_stack, all_tracks=tracks, prob_stack=prob_stack))
        y.append(1.0 if support > 0.3 else 0.0)
    if not x:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(x), np.array(y, dtype=np.float32)


def run_training(args):
    images, gts = load_stacks(args.input)
    if len(images) == 0:
        raise ValueError("No training stacks with paired masks found")
    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.pretrained_model)

    xs, ys = [], []
    use_prob = getattr(args, "train_use_prob", True)
    for i, (img, gt) in enumerate(zip(images, gts)):
        if args.simulate_z_dropout:
            gt = simulate_z_dropout(gt, dropout_prob=args.dropout_prob, seed=args.seed + i)
        pred, prob_stack = [], []
        for z in range(img.shape[0]):
            masks, flows, *_ = model.eval(img[z], do_3D=False, diameter=args.diameter)
            pred.append(masks.astype(np.int32))
            prob_stack.append(_extract_prob_from_flows(flows) if use_prob else None)
        prob_stack = None if not use_prob else np.stack([np.zeros_like(pred[0], dtype=np.float32) if p is None else p for p in prob_stack], axis=0)

        x, y = _extract_features_and_labels(img, gt, pred, prob_stack=prob_stack)
        sx, sy = _synthetic_refiner_samples(img, gt, prob_stack=prob_stack, seed=args.seed + i, n_per_obj=getattr(args, "synthetic_per_obj", 6))
        if len(x):
            xs.append(x); ys.append(y)
        if len(sx):
            xs.append(sx); ys.append(sy)

    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    if getattr(args, "refiner_nonlinear", True):
        clf = NonLinearRefiner(n_features=x_train.shape[1], hidden_dim=getattr(args, "refiner_hidden_dim", 16))
    else:
        clf = LinearRefiner(n_features=x_train.shape[1])
    losses = clf.fit(x_train, y_train, lr=args.learning_rate, epochs=args.epochs, batch_size=getattr(args, "batch_size", 64))
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
