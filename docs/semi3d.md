# Cellpose Semi-3D Z-Connectivity Mode

`semi3d` is a post-processing pipeline that runs regular 2D Cellpose per z-slice and then links objects across z to improve stack continuity without full 3D CNN inference.

## Algorithm overview

```text
Input stack -> 2D Cellpose per-slice -> instance nodes (z,id,geom)
         -> cross-slice graph linking (IoU + distance + size, gap-tolerant)
         -> mandatory missing-slice reconstruction (distance-transform prior + constrained deformation)
         -> track confidence scoring (length/link/shape/intensity/gaps)
         -> keep / bridge / recover decisions
         -> refined per-slice masks (+ optional 3D track labels)
```

## Install / availability

No extra package is required beyond Cellpose dependencies. The mode is available from the normal CLI entrypoint:

- `cellpose semi3d`
- `cellpose semi3d-train`
- `cellpose semi3d-eval`

## Inference

```bash
cellpose --semi3d \
  --semi3d_input /path/to/stack.tif \
  --semi3d_output /path/to/out \
  --semi3d_pretrained_model cpsam \
  --semi3d_max_gap 2 \
  --semi3d_min_track_len 2 \
  --semi3d_min_conf 0.05 \
  --semi3d_save_3d_labels
```

## Training (refinement classifier)

```bash
cellpose --semi3d_train \
  --semi3d_input /path/to/train_stacks \
  --semi3d_output /path/to/model_out \
  --semi3d_simulate_z_dropout \
  --semi3d_dropout_prob 0.2 \
  --semi3d_epochs 200
```

Expected input format: each image stack has paired GT as either `<stem>_masks.tif` or `<stem>_seg.npy` (GUI annotation format).

## Evaluation

```bash
cellpose --semi3d_eval \
  --semi3d_pred /path/to/semi3d_refined_masks.tif \
  --semi3d_gt /path/to/gt_masks.tif \
  --semi3d_output /path/to/eval_out
```

Inference outputs include:
- `semi3d_refined_masks.tif`: final per-slice labels.
- `semi3d_track_labels.tif` (optional): z-consistent track labels.
- `semi3d_reconstructed_flags.tif`: binary map where reconstructed (not directly detected) pixels are marked as 1.

Outputs JSON metrics including:
- Dice / IoU mean (slice-wise)
- Instance AP proxy
- Z-Continuity F1
- False Split Rate
- Recovery IoU

## CLI flags

Semi3D runs through the main Cellpose parser with mode flags:
- `--semi3d`: run inference
- `--semi3d_train`: run refinement training
- `--semi3d_eval`: run evaluation

Common Semi3D parameters:
- `--semi3d_input`, `--semi3d_output`, `--semi3d_seed`
- `--semi3d_pretrained_model`, `--semi3d_diameter`
- `--use_gpu`, `--verbose` (shared global flags)

Inference-specific:
- `--semi3d_link_iou`, `--semi3d_link_dist`, `--semi3d_size_tolerance`, `--semi3d_max_gap`
- `--semi3d_min_track_len`, `--semi3d_min_conf`, `--semi3d_save_3d_labels`

Training-specific:
- `--semi3d_simulate_z_dropout`, `--semi3d_dropout_prob`
- `--semi3d_learning_rate`, `--semi3d_epochs`

Eval-specific:
- `--semi3d_pred`, `--semi3d_gt`
