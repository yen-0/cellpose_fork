# Semi-3D Algorithm Details (Cellpose)

This document expands the high-level Semi-3D overview into an explicit step-by-step algorithm.

## 1) Inputs and outputs

### Inputs
- A z-stack image volume `I[z]` (multi-page TIFF or ordered slice directory).
- A 2D Cellpose model (for example `cpsam`) used independently on each slice.
- Optional trained refiner model `semi3d_refiner.npz`.

### Outputs
- `semi3d_refined_masks.tif`: final per-slice labels after Semi-3D reasoning.
- `semi3d_track_labels.tif` (optional): z-consistent instance IDs.
- `semi3d_reconstructed_flags.tif`: pixels reconstructed in missing slices (1=reconstructed).

---

## 2) Stage A — Independent 2D inference per slice

For each slice index `z`:
1. Run standard 2D Cellpose:
   - `masks[z], flows[z] = CellposeModel.eval(I[z], do_3D=False, ...)`
2. Extract instances from `masks[z]`:
   - Binary mask per ID
   - Area
   - Centroid

This stage deliberately does **not** use a 3D CNN.

---

## 3) Stage B — Node graph construction across z

Construct nodes:
- `node = (z, instance_id, mask, centroid, area)`

For each active track ending at `z_last`, evaluate candidates in slices up to `z_last + max_gap`.
A candidate link is valid when constraints are satisfied:
- Centroid distance <= `link_dist`
- Area ratio >= `size_tolerance`
- For adjacent slices (`gap=1`), IoU >= `link_iou`

Among valid candidates, choose the best score:

`score = IoU + 0.5*area_ratio - 0.01*distance - 0.05*(gap-1)`

Append best match to track and count skipped slices as gap bridges.

---

## 4) Stage C — Mandatory missing-slice reconstruction

For any accepted linked track with a gap (`z_a < z_missing < z_b`), reconstruction is mandatory.
No recovery threshold gates reconstruction once the link is valid.

### 4.1 Shape prior via distance-transform interpolation

Given masks `M_a` and `M_b`:
1. Compute signed distance fields (SDF):
   - `S_a = dist(outside(M_a)) - dist(inside(M_a))`
   - `S_b = dist(outside(M_b)) - dist(inside(M_b))`
2. Interpolate at normalized position `alpha = (z_missing-z_a)/(z_b-z_a)`:
   - `S_mid = (1-alpha)*S_a + alpha*S_b`
3. Initial prior mask:
   - `M_prior = (S_mid < 0)`

This preserves shape transitions better than centroid/area scaling.

### 4.2 Constrained deformation using slice evidence

Refine `M_prior` on target image slice `I[z_missing]`:
1. Compute normalized edge map (Laplacian + blur).
2. Iteratively deform contour:
   - Expand boundary where edge is weak.
   - Shrink boundary where edge is strong.
3. Apply morphology (open/close) for stability.
4. Enforce locality constraint to stay near prior (distance-to-prior bound).
5. Optionally add weak advection from Cellpose flow vectors.

Result is reconstructed mask `M_hat[z_missing]`.

### 4.3 Reconstruction flags

All reconstructed pixels are marked in `semi3d_reconstructed_flags.tif`.
This distinguishes reconstructed vs directly detected regions.

---

## 5) Stage D — Track confidence and filtering

Each track receives a confidence score:

`track_conf = length_score * mean_link_iou * shape_consistency * intensity_support * gap_penalty`

Where:
- `length_score`: longer tracks score higher
- `mean_link_iou`: consistency of linked masks
- `shape_consistency`: low area variance across z
- `intensity_support`: object intensity support in image
- `gap_penalty`: penalizes many skipped slices

Tracks are removed if:
- length < `min_track_len`, or
- confidence < `min_conf`

---

## 6) Optional learned refiner (`semi3d_refiner.npz`)

If `--semi3d_refiner_model` is provided during inference:
1. Load linear refiner weights.
2. Build the same track feature vector used in training.
3. Predict keep probability `p_keep` per track.
4. Keep only tracks with `p_keep >= semi3d_refiner_threshold`.

Then perform final relabel/reconstruction on retained tracks.

---

## 7) Training workflow summary (`--semi3d_train`)

1. Load image stacks + paired labels from either:
   - `<stem>_masks.tif/.tiff`
   - `<stem>_seg.npy`
2. Optional z-dropout simulation removes GT in random slices.
3. Run per-slice 2D Cellpose to get predicted masks.
4. Build tracks and extract feature vectors.
5. Label tracks (keep/discard surrogate target).
6. Train lightweight linear classifier.
7. Save `semi3d_refiner.npz`.

---

## 8) Evaluation workflow summary (`--semi3d_eval`)

Given predicted refined stack and GT stack:
- Slice metrics: Dice, IoU
- Instance AP proxy
- Z-Continuity F1
- False Split Rate
- Recovery IoU

Results are written to `semi3d_eval.json`.

---

## 9) Practical notes

- Semi-3D is a post-processing framework on top of 2D inference, not a full 3D neural backbone.
- The algorithm is designed to fix missing-middle slices by coupling adjacent detections in z.
- Use deterministic seed (`--semi3d_seed`) for reproducible runs.
