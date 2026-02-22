import os
import numpy as np
from cellpose import models
from .dataset import load_stacks
from .augment import simulate_z_dropout
from .model import LinearRefiner, NonLinearRefiner
from ..linking import build_association_tracks
from ..confidence import track_confidence


def extract_track_features(track, image_stack, source_masks=None):
    return np.array([
        len(track.nodes),
        np.mean(track.links) if track.links else 0.0,
        np.std([n.area for n in track.nodes]) if len(track.nodes) > 1 else 0.0,
        track.gap_bridges,
        track_confidence(track, image_stack, source_masks=source_masks),
    ], dtype=np.float32)


def _extract_features_and_labels(image_stack, gt_stack, pred_stack):
    tracks = build_association_tracks(pred_stack)
    x, y = [], []
    gt_present = np.array([np.any(g > 0) for g in gt_stack], dtype=np.float32)
    for tr in tracks:
        zmin, zmax = tr.nodes[0].z, tr.nodes[-1].z
        support = float(gt_present[zmin:zmax + 1].mean())
        x.append(extract_track_features(tr, image_stack, source_masks=pred_stack))
        y.append(1.0 if support > 0.3 else 0.0)
    if not x:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32)
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
    clf.fit(x_train, y_train, lr=args.learning_rate, epochs=args.epochs)
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "semi3d_refiner.npz")
    clf.save(out_path)
    return out_path, len(x_train)
