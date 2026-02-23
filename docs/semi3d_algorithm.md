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


## 10) Linking implementation notes (current behavior)

In the current code path, linking is **streamed per z-slice** (not all cells at once):

1. For the current slice `z`, extract instance nodes only for that slice.
2. Keep an `active` list of tracks that can still receive links (within `max_gap`).
3. For each active track, score candidates in the current slice and take the best valid match.
4. Start new tracks for unmatched nodes.
5. Drop tracks that are too old to receive future links.

Node representation is compact to reduce memory:
- bounding box (`bbox`)
- cropped binary mask (`mask_crop`)
- centroid
- area

IoU is computed only in overlapping bbox windows, not full-frame mask arrays.
This is much cheaper than materializing full HxW masks for every candidate pair.

---

## 11) Why linking still gets expensive

Even with streaming, runtime/memory can still rise when:
- each slice has many instances,
- many tracks remain active at once,
- `link_dist` is large (more candidate pairs survive),
- `max_gap` is large (active set persists longer).

In those regimes, complexity is dominated by pair scoring:

`O(sum_z [#active_tracks(z) * #nodes(z)])`

So the main wins come from **candidate pruning** and **faster pair scoring**.

---

## 12) High-impact efficiency improvements (recommended)

### A) Spatial indexing / gated candidate search (CPU)

Before IoU, query only nearby candidates using a spatial index (grid hash or KD-tree on centroids).
This changes practical behavior from all-pairs to local-neighborhood pairs.

- Build index per slice once.
- For each active track, query radius = `link_dist`.
- Evaluate IoU/area only on returned neighbors.

This is usually the largest immediate speedup.

### B) Two-stage scoring cascade

Use cheap filters first, expensive checks last:
1. centroid distance gate
2. area-ratio gate
3. bbox-overlap gate
4. IoU on crop overlap

Many pairs die early, reducing total IoU calls.

### C) Track-state compaction

For older nodes in a track, keep summary stats only.
Retain full crop mask for the most recent node(s) used in linking.
This cuts memory for long tracks.

### D) Block/chunk processing in z

Process z in windows (for example 16–64 slices), serialize intermediate track state, and continue.
This bounds peak memory for very deep stacks.

### E) Optional approximation mode

For very dense data, use bbox-IoU surrogate first and compute exact mask IoU only for top-k candidates.

---

## 13) GPU acceleration opportunities for linking

Right now GPU is used for:
- stage1 Cellpose inference (`--use_gpu`)
- optional stage2 refiner scoring (`--semi3d_stage2_use_gpu`)

Linking itself is still CPU-oriented. To harness GPU for linking:

### Option 1: GPU candidate scoring with torch tensors

- Pack active centroids and current-slice centroids into tensors.
- Compute full distance matrix on GPU in one shot.
- Apply distance/area gating with tensor masks.
- Return only surviving candidate indices to CPU for final IoU.

Benefit: massive speedup in dense slices with many objects.

### Option 2: GPU batched IoU on cropped masks

For surviving pairs:
- pad/pack crops by size groups,
- compute intersections/unions in batched torch ops,
- keep top-1 per active track.

This moves the expensive boolean math to GPU.

### Option 3: Hybrid strategy (recommended first)

- GPU for distance + area prefiltering,
- CPU for exact crop-IoU,
- optional GPU path behind a flag.

This gives most benefit with low implementation risk.

---

## 14) Practical tuning tips

If you need speed right now without major code changes:
- reduce `link_dist` until recall drops,
- keep `max_gap` only as large as biologically needed,
- raise `size_tolerance` slightly,
- run stage1/stage2 split and inspect stage1 masks to reduce false instances early.

These directly shrink candidate count and linking cost.


## 15) Implemented optimizations in current code

The current implementation now includes the first three CPU optimizations plus hybrid GPU prefiltering:
- **A) Spatial indexing**: grid-based centroid neighborhood query per slice.
- **B) Two-stage scoring cascade**: distance -> area ratio -> bbox overlap -> IoU.
- **C) Track-state compaction**: older linked nodes drop dense `mask_crop` tensors.
- **Hybrid GPU option (Option 3)**: `--semi3d_link_gpu_prefilter` runs distance/size candidate gating on GPU (when CUDA is available), then keeps exact IoU/link decisions on CPU.
