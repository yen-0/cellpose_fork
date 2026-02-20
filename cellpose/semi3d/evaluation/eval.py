import os
import json
import numpy as np
from cellpose import io
from .metrics import dice_score, iou_score, z_continuity_f1, false_split_rate, recovery_iou


def run_evaluation(args):
    pred = io.imread(args.pred)
    gt = io.imread(args.gt)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT stacks must have matching shape")

    dice = [dice_score(pred[z], gt[z]) for z in range(pred.shape[0])]
    iou = [iou_score(pred[z], gt[z]) for z in range(pred.shape[0])]

    # synthetic removed-slice recovery metric
    z_mid = pred.shape[0] // 2
    rec_iou = recovery_iou(gt[z_mid], pred[z_mid])

    metrics = {
        "dice_mean": float(np.mean(dice)),
        "iou_mean": float(np.mean(iou)),
        "instance_ap": float(np.mean([d > 0.5 for d in iou])),
        "z_continuity_f1": float(z_continuity_f1(pred)),
        "false_split_rate": float(false_split_rate(pred)),
        "recovery_iou": float(rec_iou),
    }

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "semi3d_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path, metrics
