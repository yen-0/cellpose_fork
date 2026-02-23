# Cellpose Semi-3D Z-Connectivity Mode

For a full step-by-step algorithm description, see [Semi-3D Algorithm Details](semi3d_algorithm.md).

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
  --semi3d_refiner_model /path/to/model_out/semi3d_refiner.npz \
  --semi3d_refiner_threshold 0.5 \
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

After training, use the generated `semi3d_refiner.npz` during inference with `--semi3d_refiner_model` to apply the learned keep/discard filter. Training also saves `semi3d_refiner_train_log.json` with per-epoch loss and best epoch.

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
- `--semi3d_refiner_model`, `--semi3d_refiner_threshold`
- `--semi3d_use_prob_occupancy`, `--semi3d_prob_occupancy_thresh`
- stage1 candidate/defog knobs: `--semi3d_stage1_cellprob_threshold`, `--semi3d_stage1_masks_from_defog_prob`, `--semi3d_stage1_mask_prob_threshold`, `--semi3d_stage1_mask_min_area`, `--semi3d_disable_stage1_prob_defog`, `--semi3d_stage1_prob_bg_sigma`, `--semi3d_stage1_prob_bg_percentile`, `--semi3d_stage1_prob_hi_percentile`, `--semi3d_stage1_prob_gamma`

Training-specific:
- `--semi3d_simulate_z_dropout`, `--semi3d_dropout_prob`
- `--semi3d_learning_rate`, `--semi3d_epochs`, `--semi3d_synthetic_per_obj`, `--semi3d_refiner_batch_size`

Eval-specific:
- `--semi3d_pred`, `--semi3d_gt`

## Proposed improvements (planning)

The current Semi-3D pipeline is performing well, but there are two high-impact opportunities:

### 1) Reduce memory footprint without changing outputs

- **Stream slices in chunks instead of holding full-stack intermediates**: process z in windows (for example 16–64 slices), carry only boundary track state between windows, and flush temporary per-slice tensors/masks after linking.
- **Store sparse object features, not full dense masks, during association**: keep bbox, centroid, area, and compact run-length encoding (RLE) for candidates; only materialize dense masks when reconstructing or writing final outputs.
- **Use low-precision buffers where safe**: convert non-critical float arrays (distance/intensity helper maps) to `float16` and labels to minimal integer dtypes (`uint16`/`int32` fallback).
- **Lazy-write optional outputs**: gate and defer expensive arrays (`semi3d_track_labels.tif`, reconstruction flags) behind explicit flags and write incrementally slice-by-slice.
- **Memory profiling checkpoints in pipeline stages**: add timing + RSS/VRAM snapshots around segmentation, linking, reconstruction, and refinement to identify the dominant peak before/after each optimization.

### 2) Count and score two-slice skips explicitly

- **Promote gap-length awareness from binary to multi-class**: treat links as `direct (gap=0)`, `single-skip (gap=1)`, and `double-skip (gap=2)` rather than only rewarding one skipped slice.
- **Update confidence features**: add dedicated counters and ratios for `gap=1` and `gap=2` bridges so double-skip recovery contributes positively when geometrically consistent.
- **Rebalance penalties by skip length instead of hard rejection**: allow `gap=2` links with stricter IoU/shape constraints and stronger (but finite) penalty, instead of effectively dropping them.
- **Train refiner with synthetic 2-skip dropout examples**: extend z-dropout simulation to include contiguous two-slice removals so the learned model recognizes valid longer-gap recoveries.
- **Expose skip-aware metrics in evaluation**: report precision/recall for recovered `gap=1` and `gap=2` events separately, plus a weighted continuity score to reflect practical improvements.

### Suggested execution order

1. Add skip-aware metrics and diagnostics first (so improvements are measurable).
2. Implement chunked processing + sparse representation for memory reduction.
3. Add explicit `gap=2` link scoring and training augmentation.
4. Re-tune thresholds using eval outputs split by gap length.


## Two-stage execution

You can now run Semi-3D in two explicit endpoints:

1. `--semi3d_stage1`: run per-slice Cellpose inference, optional border instance cleanup, and save intermediate masks.
2. `--semi3d_stage2`: load stage1 masks and run linking/reconstruction/refinement.

Useful flags:
- `--semi3d_stage1_masks` (preferred TIFF), `--semi3d_stage1_flows`, `--semi3d_stage1_prob`
- `--semi3d_memmap_stage2_inputs` (memory-map masks in stage2)
- `--semi3d_stage2_use_gpu` (GPU refiner scoring in stage2)
- `--semi3d_link_gpu_prefilter` (hybrid GPU prefilter for linking)
- `--semi3d_merge_dist` (merge split detections of the same cell within a slice)
- `--semi3d_border_exclusion_px`
- `--semi3d_fill_edges` (fills first/last slices too)
- `--semi3d_allow_overlap_recon` (disable occupancy-aware reconstruction blocking)
- `--semi3d_recon_min_free_fraction` (reject impossible reconstructions that mostly overlap established cells)
- `--semi3d_use_prob_occupancy`, `--semi3d_prob_occupancy_thresh` (use stage1 probability map as occupancy prior in stage2)
- `--semi3d_save_debug_tiff` (save debug flow/prob/refiner maps as TIFF)
- `--semi3d_max_gap` (supports 2+ skips)
- `--use_gpu` for accelerated stage1 inference
- `--semi3d_refiner_linear` to force linear model; nonlinear refiner is default


### Filling first and last slices

Use `--semi3d_fill_edges` during `--semi3d` or `--semi3d_stage2` to run boundary reconstruction on first/last slices using nearest valid track masks as priors (with image/flow refinement).


### Stage2 launching note

`--semi3d_stage2` is a true semi3d mode and is dispatched through the semi3d runner (not GUI mode).


### Refiner caveat update

Refiner features now include overlap-conflict signals between tracks (same-slice overlap ratio / conflict ratio) to better down-rank impossible masks in crowded regions.


### Occupancy safety behavior

With default settings, both direct track assignments and reconstructed masks are clipped to free pixels only, so already-confirmed territory is not overwritten. Tracks are applied in descending confidence order.


### Refiner retraining note

If you change linking/reconstruction settings (merge distance, overlap/occupancy rules, gap behavior), retraining the refiner is strongly recommended because feature distributions shift.


### Strict occupancy rule

With default behavior, reconstruction is forbidden from writing into territories already claimed by linked track assignments. Direct linked claims are locked first, then reconstruction is applied only on remaining free pixels.


### Refiner training data generation

Training now augments GT with synthetic distorted/displaced target masks and overlap-with-other-cell conflict samples (impossible territory examples) so the refiner learns to reject placements that invade already-occupied regions.


## Exact commands (copy/paste)

### Stage1
```bash
cellpose --semi3d_stage1 \
  --semi3d_input /path/to/stack.tif \
  --semi3d_output /path/to/out \
  --semi3d_pretrained_model cpsam \
  --semi3d_stage1_cellprob_threshold -2.0 \
  --semi3d_stage1_masks_from_defog_prob --semi3d_stage1_mask_prob_threshold 0.35 --semi3d_stage1_mask_min_area 20 \
  --semi3d_stage1_prob_bg_sigma 40 --semi3d_stage1_prob_bg_percentile 5 --semi3d_stage1_prob_hi_percentile 97 --semi3d_stage1_prob_gamma 0.9 \
  --use_gpu --verbose
```

### Stage2 with strict occupancy + probability prior
```bash
cellpose --semi3d_stage2 \
  --semi3d_input /path/to/stack.tif \
  --semi3d_output /path/to/out \
  --semi3d_stage1_masks /path/to/out/semi3d_stage1_masks.tif \
  --semi3d_stage1_prob /path/to/out/semi3d_stage1_prob.tif \
  --semi3d_fill_edges \
  --semi3d_use_prob_occupancy --semi3d_prob_occupancy_thresh 0.5 \
  --semi3d_recon_min_free_fraction 0.25 \
  --semi3d_refiner_model /path/to/model_out/semi3d_refiner.npz \
  --semi3d_refiner_threshold 0.5 \
  --semi3d_save_debug_tiff --verbose
```

### Refiner training (prob-on by default, minibatch)
```bash
cellpose --semi3d_train \
  --semi3d_input /path/to/train_stacks \
  --semi3d_output /path/to/model_out \
  --semi3d_pretrained_model cpsam \
  --semi3d_simulate_z_dropout --semi3d_dropout_prob 0.2 \
  --semi3d_synthetic_per_obj 6 --semi3d_refiner_batch_size 64 \
  --semi3d_epochs 200 --use_gpu --verbose
```

