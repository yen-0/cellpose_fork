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
cellpose semi3d \
  --input /path/to/stack.tif \
  --output /path/to/out \
  --pretrained_model cpsam \
  --max-gap 2 \
  --min-track-len 2 \
  --min-conf 0.05 \
  --save-3d-labels
```

## Training (refinement classifier)

```bash
cellpose semi3d-train \
  --input /path/to/train_stacks \
  --output /path/to/model_out \
  --simulate-z-dropout \
  --dropout-prob 0.2 \
  --epochs 200
```

Expected input format: each image stack has paired GT mask stack named `<stem>_masks.tif`.

## Evaluation

```bash
cellpose semi3d-eval \
  --pred /path/to/semi3d_refined_masks.tif \
  --gt /path/to/gt_masks.tif \
  --output /path/to/eval_out
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

### `semi3d`
- `--input`: stack file (`.tif/.tiff`) or ordered slice directory.
- `--output`: output directory.
- `--pretrained_model`, `--use_gpu`, `--diameter`, `--seed`.
- `--link-iou`: minimum overlap for adjacent linking.
- `--link-dist`: max centroid distance for linking.
- `--size-tolerance`: minimum relative area similarity.
- `--max-gap`: max z gap allowed when linking tracks.
- Missing slices inside valid linked tracks are always reconstructed (no recovery/bridge gating threshold).
- `--min-track-len`: minimum linked length to keep a track.
- `--min-conf`: minimum final track confidence.
- `--save-3d-labels`: save tracked instance labels across z.

### `semi3d-train`
- `--input`, `--output`, `--pretrained_model`, `--use_gpu`, `--diameter`, `--seed`.
- `--simulate-z-dropout`: enable simulated z-dropout augmentation.
- `--dropout-prob`: probability of deleting GT masks per slice.
- `--learning-rate`, `--epochs`: training hyperparameters for lightweight linear refiner.

### `semi3d-eval`
- `--pred`: predicted refined stack.
- `--gt`: ground-truth stack.
- `--output`: output directory for metric JSON.
- `--seed`: deterministic seed.
