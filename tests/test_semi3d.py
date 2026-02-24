import numpy as np

from cellpose.semi3d.linking import build_association_tracks
from cellpose.semi3d.refine import relabel_tracks
from cellpose.semi3d.training import dataset
from cellpose.cli import get_arg_parser




def test_first_slice_masks_are_never_dropped():
    s0 = np.zeros((48, 48), dtype=np.int32)
    s1 = np.zeros((48, 48), dtype=np.int32)
    s0[5:10, 5:10] = 1
    s0[20:26, 20:26] = 2
    s0[35:40, 8:13] = 3
    # second slice can be empty; first-slice tracks must still be preserved.

    tracks = build_association_tracks([s0, s1], link_iou=0.5, link_dist=5.0, max_gap=2)
    assert len(tracks) >= 3
    starts = sorted([tr.nodes[0].instance_id for tr in tracks[:3]])
    assert starts == [1, 2, 3]



def test_relabel_keeps_first_slice_tracks_even_below_thresholds():
    from cellpose.semi3d.linking import Track, InstanceNode
    from cellpose.semi3d.refine import relabel_tracks

    shape = (16, 16)
    m0 = np.zeros(shape, dtype=bool); m0[3:7, 3:7] = True
    n0 = InstanceNode(z=0, instance_id=1, bbox=(3,7,3,7), centroid=np.array([4.5,4.5]), area=int(m0.sum()), mask_crop=m0[3:7,3:7])
    t0 = Track(track_id=1, nodes=[n0], links=[])

    img = np.stack([np.zeros(shape, dtype=np.float32)], axis=0)
    out, _, _, kept = relabel_tracks([t0], img, min_track_len=3, min_conf=0.99)

    assert len(kept) == 1
    assert np.any(out[0] > 0)
def test_gap_tolerant_linking_and_mandatory_reconstruction():
    s0 = np.zeros((32, 32), dtype=np.int32)
    s1 = np.zeros((32, 32), dtype=np.int32)
    s2 = np.zeros((32, 32), dtype=np.int32)
    s0[10:14, 10:14] = 1
    s2[11:15, 11:15] = 1

    tracks = build_association_tracks([s0, s1, s2], link_iou=0.0, link_dist=10.0, max_gap=3)
    assert len(tracks) == 1

    img = np.stack([s0.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32)], axis=0)
    refined, _, reconstructed_flags, kept = relabel_tracks(tracks, img, min_track_len=1, min_conf=0.0)
    assert len(kept) == 1
    assert np.any(refined[1] > 0)
    assert np.any(reconstructed_flags[1] > 0)


def test_two_skip_linking_supported():
    slices = [np.zeros((32, 32), dtype=np.int32) for _ in range(5)]
    slices[0][10:14, 10:14] = 1
    slices[3][10:14, 10:14] = 1
    tracks = build_association_tracks(slices, link_iou=0.0, link_dist=10.0, max_gap=4)
    assert len(tracks) == 1
    assert tracks[0].gap_hist.get(2, 0) == 1






def test_merge_requires_iou_improvement_only():
    s0 = np.zeros((48, 48), dtype=np.int32)
    s1 = np.zeros((48, 48), dtype=np.int32)
    s0[16:24, 16:24] = 1
    s1[16:24, 16:20] = 2
    s1[16:24, 20:24] = 3

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=14.0, max_gap=2)
    assert len(tracks) == 1
    assert len(tracks[0].nodes) == 2
    assert tracks[0].nodes[-1].area >= 60



def test_matching_prioritizes_highest_iou_a_over_iou_b():
    s0 = np.zeros((32, 32), dtype=np.int32)
    s1 = np.zeros((32, 32), dtype=np.int32)

    # previous mask (A): 6x6 block
    s0[10:16, 10:16] = 1
    # candidate 2: exact overlap + extra pixels -> IoU A = 1.0, IoU B < 1
    s1[10:16, 10:16] = 2
    s1[16:18, 10:16] = 2
    # candidate 3: strict subset -> IoU B = 1.0, IoU A lower
    s1[10:14, 10:14] = 3

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=10.0, max_gap=2)
    assert len(tracks) >= 2

    linked = [tr for tr in tracks if len(tr.nodes) == 2]
    assert len(linked) == 1
    assert linked[0].nodes[1].instance_id == 2


def test_merge_uses_iou_b_signal():
    s0 = np.zeros((48, 48), dtype=np.int32)
    s1 = np.zeros((48, 48), dtype=np.int32)

    # previous mask A is wide.
    s0[16:24, 16:32] = 1
    # node 2 covers left part (not enough IoU B to trigger merge).
    s1[16:24, 16:24] = 2
    # node 3 covers right part and should be merged based on IoU B evidence.
    s1[16:24, 24:32] = 3

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=20.0, max_gap=2, merge_dist=16.0)
    assert len(tracks) == 1
    assert len(tracks[0].nodes) == 2
    # merged node should recover full previous area
    assert tracks[0].nodes[-1].area == 128

def test_forced_link_attaches_leftover_with_tiny_iou():
    s0 = np.zeros((64, 64), dtype=np.int32)
    s1 = np.zeros((64, 64), dtype=np.int32)
    s0[8:14, 8:14] = 1
    # no-overlap and fairly far, but within 2*link_dist fallback
    s1[22:28, 8:14] = 2

    tracks = build_association_tracks([s0, s1], link_iou=0.5, link_dist=10.0, max_gap=2)
    assert len(tracks) == 1
    assert len(tracks[0].nodes) == 2

def test_edge_fill_reconstructs_first_and_last_slices():
    s0 = np.zeros((32, 32), dtype=np.int32)
    s1 = np.zeros((32, 32), dtype=np.int32)
    s2 = np.zeros((32, 32), dtype=np.int32)
    s3 = np.zeros((32, 32), dtype=np.int32)
    s1[8:12, 8:12] = 1
    s2[8:12, 8:12] = 1
    tracks = build_association_tracks([s0, s1, s2, s3], link_iou=0.0, link_dist=10.0, max_gap=3)
    img = np.stack([s.astype(np.float32) for s in [s0, s1, s2, s3]], axis=0)
    refined, _, flags, kept = relabel_tracks(tracks, img, min_track_len=1, min_conf=0.0, fill_edges=True)
    assert len(kept) == 1
    assert np.any(refined[0] > 0)
    assert np.any(refined[3] > 0)
    assert np.any(flags[0] > 0)
    assert np.any(flags[3] > 0)




def test_leftover_prefers_overlap_or_area_radius_before_new_track():
    s0 = np.zeros((64, 64), dtype=np.int32)
    s1 = np.zeros((64, 64), dtype=np.int32)

    # active track from first slice
    s0[8:14, 8:14] = 1
    # leftover node with tiny overlap (single pixel) should still attach
    s1[13:19, 13:19] = 2

    tracks = build_association_tracks([s0, s1], link_iou=0.9, link_dist=2.0, max_gap=2)
    assert len(tracks) == 1
    assert len(tracks[0].nodes) == 2


def test_leftover_falls_back_to_nearest_active_track_not_new_track():
    s0 = np.zeros((96, 96), dtype=np.int32)
    s1 = np.zeros((96, 96), dtype=np.int32)

    s0[10:16, 10:16] = 1
    s0[70:76, 70:76] = 2
    # New node far from both with no overlap and outside 2*area radius gate
    s1[40:46, 40:46] = 3

    tracks = build_association_tracks([s0, s1], link_iou=0.9, link_dist=2.0, max_gap=2)
    # Should connect to some existing graph instead of creating a third one
    assert len(tracks) == 2
    assert sorted(len(t.nodes) for t in tracks) == [1, 2]





def test_primary_prefers_previous_slice_before_two_slice_history():
    slices = [np.zeros((64, 64), dtype=np.int32) for _ in range(3)]
    # Track A at z0 and z1, slightly shifted
    slices[0][20:26, 20:26] = 1
    slices[1][22:28, 22:28] = 1
    # Track B only at z0 near the same area (history-only competitor at z2)
    slices[0][24:30, 24:30] = 2
    # z2 candidate should link using z1 anchor first, not z0-only history track
    slices[2][22:28, 22:28] = 3

    tracks = build_association_tracks(slices, link_iou=0.0, link_dist=12.0, max_gap=3)

    t_primary = None
    for t in tracks:
        if t.nodes and t.nodes[0].z == 0 and t.nodes[0].instance_id == 1:
            t_primary = t
            break
    assert t_primary is not None
    zs = [n.z for n in t_primary.nodes]
    assert 2 in zs

def test_links_back_using_two_to_three_slice_history_when_prev_missing():
    slices = [np.zeros((48, 48), dtype=np.int32) for _ in range(4)]
    # z0 object exists
    slices[0][20:26, 20:26] = 1
    # z1 object missing (dropout)
    # z2 object reappears and should reconnect to original track via history
    slices[2][21:27, 21:27] = 2

    tracks = build_association_tracks(slices, link_iou=0.0, link_dist=8.0, max_gap=3)

    linked = [t for t in tracks if len(t.nodes) >= 2]
    assert len(linked) == 1
    zs = [n.z for n in linked[0].nodes]
    assert 0 in zs and 2 in zs
    assert linked[0].gap_hist.get(1, 0) >= 1


def test_history_anchor_can_beat_last_node_for_reappearance():
    slices = [np.zeros((80, 80), dtype=np.int32) for _ in range(4)]
    # Track A exists at z0 near (10,10), then drifts away at z1
    slices[0][8:14, 8:14] = 1
    slices[1][48:54, 48:54] = 3
    # Track B at z1 near reappearance location
    slices[1][10:16, 10:16] = 2
    # reappearance at z2 should use recent history (z0 anchor) and link back to track A
    slices[2][9:15, 9:15] = 4

    tracks = build_association_tracks(slices, link_iou=0.0, link_dist=8.0, max_gap=3)

    # find track that started from z0 id=1
    t0 = None
    for t in tracks:
        if t.nodes and t.nodes[0].z == 0 and t.nodes[0].instance_id == 1:
            t0 = t
            break
    assert t0 is not None
    zs = [n.z for n in t0.nodes]
    assert 2 in zs







def test_merge_bypassed_when_hanging_graphs_exist():
    s0 = np.zeros((96, 96), dtype=np.int32)
    s1 = np.zeros((96, 96), dtype=np.int32)

    # graph A and B in previous slice
    s0[20:30, 20:30] = 1
    s0[60:70, 60:70] = 2

    # only graph A has direct continuation; graph B remains hanging
    s1[20:30, 20:30] = 3
    # nearby extra candidate for potential merge into graph A
    s1[20:30, 31:41] = 4

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=18.0, max_gap=3, merge_dist=20.0)

    # graph A should not merge candidate 4 while graph B is still hanging
    tA = None
    for t in tracks:
        if t.nodes and t.nodes[0].instance_id == 1:
            tA = t
            break
    assert tA is not None
    assert len(tA.nodes) == 2
    assert tA.nodes[-1].area == 100

def test_merge_deferred_when_other_graph_can_attach_candidate():
    s0 = np.zeros((80, 80), dtype=np.int32)
    s1 = np.zeros((80, 80), dtype=np.int32)

    # two existing graphs on previous slice
    s0[20:30, 20:30] = 1
    s0[20:30, 35:45] = 2

    # candidate chosen by graph 1
    s1[20:30, 20:30] = 3
    # high-IoU_B candidate for graph 1, but also a strong direct attach for graph 2
    s1[20:30, 35:45] = 4

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=20.0, max_gap=2, merge_dist=25.0)

    # both graphs should attach separately; candidate 4 should not be merged into graph 1
    assert len(tracks) == 2
    assert all(len(t.nodes) == 2 for t in tracks)
    assert sorted(t.nodes[-1].area for t in tracks) == [100, 100]

def test_merge_chooses_single_highest_iou_b_above_threshold():
    s0 = np.zeros((64, 64), dtype=np.int32)
    s1 = np.zeros((64, 64), dtype=np.int32)

    # anchor mask
    s0[20:30, 20:30] = 1
    # selected node candidate
    s1[20:30, 20:25] = 2
    # merge candidate A: IoU_B = 1.0 against anchor overlap
    s1[20:30, 25:30] = 3
    # merge candidate B: tiny overlap -> IoU_B <= 0.1 (must not be picked)
    s1[29:39, 30:40] = 4

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=20.0, max_gap=2, merge_dist=20.0)
    assert len(tracks) == 1
    assert len(tracks[0].nodes) == 2
    # merged with only the best IoU_B candidate (id=3), recovering full 10x10 anchor footprint
    assert tracks[0].nodes[-1].area == 100


def test_merge_requires_higher_iou_b_threshold_and_distance_gate():
    s0 = np.zeros((80, 80), dtype=np.int32)
    s1 = np.zeros((80, 80), dtype=np.int32)

    s0[20:30, 20:30] = 1
    s1[20:30, 20:26] = 2
    # overlap with anchor is only 10x2 => IoU_B = 20/100 = 0.2, not strictly above threshold
    s1[20:30, 26:36] = 3

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=18.0, max_gap=2, merge_dist=24.0)
    # no merge should happen at equality to threshold (must be strictly > 0.2)
    assert len(tracks) == 2
    assert sorted(len(t.nodes) for t in tracks) == [1, 2]


def test_merge_rejects_far_candidate_even_with_high_iou_b():
    s0 = np.zeros((120, 120), dtype=np.int32)
    s1 = np.zeros((120, 120), dtype=np.int32)

    s0[20:30, 20:30] = 1
    s1[20:30, 20:30] = 2
    # perfect IoU_B against previous anchor but placed far from base node centroid
    s1[50:60, 50:60] = 3

    tracks = build_association_tracks([s0, s1], link_iou=0.0, link_dist=8.0, max_gap=2, merge_dist=40.0)
    # merge distance gate should block this merge; far node becomes separate track
    assert len(tracks) == 2
    assert sorted(len(t.nodes) for t in tracks) == [1, 2]

def test_training_loader_accepts_seg_npy(monkeypatch):
    fake_dir = "/data"

    def fake_listdir(_):
        return ["stack1.tif", "stack2.tif", "stack1_seg.npy"]

    def fake_exists(path):
        return path.endswith("stack1_seg.npy")

    def fake_imread(path):
        return path

    monkeypatch.setattr(dataset.os, "listdir", fake_listdir)
    monkeypatch.setattr(dataset.os.path, "exists", fake_exists)
    monkeypatch.setattr(dataset.io, "imread", fake_imread)

    images, labels = dataset.load_stacks(fake_dir)
    assert images == ["/data/stack1.tif"]
    assert labels == ["/data/stack1_seg.npy"]


def test_main_parser_accepts_semi3d_train_flags():
    parser = get_arg_parser()
    args = parser.parse_args([
        "--semi3d_train", "--semi3d_input", "/tmp/in", "--semi3d_output", "/tmp/out",
        "--use_gpu", "--verbose", "--semi3d_simulate_z_dropout"
    ])
    assert args.semi3d_train is True
    assert args.use_gpu is True
    assert args.verbose is True
    assert args.semi3d_simulate_z_dropout is True


def test_main_parser_accepts_refiner_inference_flags():
    parser = get_arg_parser()
    args = parser.parse_args([
        "--semi3d", "--semi3d_input", "/tmp/in.tif", "--semi3d_output", "/tmp/out",
        "--semi3d_refiner_model", "/tmp/semi3d_refiner.npz", "--semi3d_refiner_threshold", "0.6",
        "--semi3d_stage1_flows", "/tmp/semi3d_stage1_flows.npy"
    ])
    assert args.semi3d is True
    assert args.semi3d_refiner_model.endswith("semi3d_refiner.npz")
    assert abs(args.semi3d_refiner_threshold - 0.6) < 1e-6


def test_main_parser_accepts_stage_flags():
    parser = get_arg_parser()
    args = parser.parse_args([
        "--semi3d_stage2", "--semi3d_input", "/tmp/in.tif", "--semi3d_output", "/tmp/out",
        "--semi3d_stage1_masks", "/tmp/semi3d_stage1_masks.tif", "--semi3d_fill_edges",
        "--semi3d_memmap_stage2_inputs", "--semi3d_stage2_use_gpu", "--semi3d_link_gpu_prefilter", "--semi3d_merge_dist", "10", "--semi3d_recon_min_free_fraction", "0.3", "--semi3d_save_debug_tiff", "--semi3d_use_prob_occupancy", "--semi3d_prob_occupancy_thresh", "0.55", "--semi3d_stage1_prob_bg_percentile", "30", "--semi3d_stage1_prob_hi_percentile", "98", "--semi3d_stage1_prob_gamma", "1.1", "--semi3d_stage1_prob_bg_sigma", "42", "--semi3d_stage1_cellprob_threshold", "-2.5", "--semi3d_stage1_prob_boundary_sigma", "1.8", "--semi3d_stage1_prob_boundary_strength", "0.45", "--semi3d_refiner_batch_size", "32"
    ])
    assert args.semi3d_stage2 is True
    assert args.semi3d_fill_edges is True
    assert args.semi3d_memmap_stage2_inputs is True
    assert args.semi3d_stage2_use_gpu is True
    assert args.semi3d_link_gpu_prefilter is True
    assert abs(args.semi3d_merge_dist - 10.0) < 1e-6
    assert abs(args.semi3d_recon_min_free_fraction - 0.3) < 1e-6
    assert args.semi3d_save_debug_tiff is True
    assert args.semi3d_use_prob_occupancy is True
    assert abs(args.semi3d_prob_occupancy_thresh - 0.55) < 1e-6
    assert abs(args.semi3d_stage1_prob_bg_percentile - 30.0) < 1e-6
    assert abs(args.semi3d_stage1_prob_hi_percentile - 98.0) < 1e-6
    assert abs(args.semi3d_stage1_prob_gamma - 1.1) < 1e-6
    assert abs(args.semi3d_stage1_prob_bg_sigma - 42.0) < 1e-6
    assert abs(args.semi3d_stage1_cellprob_threshold + 2.5) < 1e-6
    assert abs(args.semi3d_stage1_prob_boundary_sigma - 1.8) < 1e-6
    assert abs(args.semi3d_stage1_prob_boundary_strength - 0.45) < 1e-6
    assert args.semi3d_refiner_batch_size == 32


def test_relabel_does_not_overwrite_confirmed_territory():
    from cellpose.semi3d.linking import Track, InstanceNode
    from cellpose.semi3d.refine import relabel_tracks

    shape = (16, 16)
    m1 = np.zeros(shape, dtype=bool); m1[4:8, 4:8] = True
    m2 = np.zeros(shape, dtype=bool); m2[5:9, 5:9] = True
    n1 = InstanceNode(z=0, instance_id=1, bbox=(4,8,4,8), centroid=np.array([5.5,5.5]), area=int(m1.sum()), mask_crop=m1[4:8,4:8])
    n2 = InstanceNode(z=0, instance_id=2, bbox=(5,9,5,9), centroid=np.array([6.5,6.5]), area=int(m2.sum()), mask_crop=m2[5:9,5:9])
    t1 = Track(track_id=1, nodes=[n1], links=[0.9])
    t2 = Track(track_id=2, nodes=[n2], links=[0.8])

    img = np.stack([np.zeros(shape, dtype=np.float32)], axis=0)
    out, _, _, _ = relabel_tracks([t1, t2], img, min_track_len=1, min_conf=0.0, avoid_occupied=True)
    # no pixel can belong to both ids; occupancy is first-come by confidence ordering
    assert np.all((out[0] == 1) | (out[0] == 2) | (out[0] == 0))
    assert np.sum(out[0] == 1) > 0
    assert np.sum(out[0] == 2) > 0


def test_reconstruction_never_overwrites_direct_claims():
    from cellpose.semi3d.linking import Track, InstanceNode
    from cellpose.semi3d.refine import relabel_tracks

    shape = (24, 24)
    # Track A has direct detections on z0 and z2 and reconstruction in z1
    a0 = np.zeros(shape, dtype=bool); a0[8:12, 8:12] = True
    a2 = np.zeros(shape, dtype=bool); a2[8:12, 8:12] = True
    # Track B has direct detection in middle slice that overlaps A reconstruction territory
    b1 = np.zeros(shape, dtype=bool); b1[8:12, 8:12] = True

    n_a0 = InstanceNode(z=0, instance_id=1, bbox=(8,12,8,12), centroid=np.array([9.5,9.5]), area=int(a0.sum()), mask_crop=a0[8:12,8:12])
    n_a2 = InstanceNode(z=2, instance_id=1, bbox=(8,12,8,12), centroid=np.array([9.5,9.5]), area=int(a2.sum()), mask_crop=a2[8:12,8:12])
    n_b1 = InstanceNode(z=1, instance_id=2, bbox=(8,12,8,12), centroid=np.array([9.5,9.5]), area=int(b1.sum()), mask_crop=b1[8:12,8:12])

    tA = Track(track_id=1, nodes=[n_a0, n_a2], links=[1.0])
    tB = Track(track_id=2, nodes=[n_b1], links=[1.0])

    img = np.stack([np.zeros(shape, dtype=np.float32) for _ in range(3)], axis=0)
    out, _, rec_flags, _ = relabel_tracks([tA, tB], img, min_track_len=1, min_conf=0.0, avoid_occupied=True)

    # Middle slice must keep direct claim; reconstruction cannot overwrite it
    mid = out[1]
    vals = np.unique(mid[8:12, 8:12])
    vals = vals[vals > 0]
    assert len(vals) == 1
    # reconstruction flags in overlapped region should be zero due to occupancy lock
    assert not np.any(rec_flags[1][8:12, 8:12])


def test_stable_label_colormap_reduces_duplicates_beyond_60():
    from cellpose.semi3d.cli import _stable_label_colormap

    cmap = _stable_label_colormap(128)
    unique = np.unique(cmap[1:], axis=0)
    assert unique.shape[0] >= 120


def test_labels_to_rgb_is_deterministic():
    from cellpose.semi3d.cli import _labels_to_rgb

    labels = np.zeros((3, 16, 16), dtype=np.int32)
    labels[0, 2:6, 2:6] = 1
    labels[1, 4:8, 4:8] = 61
    labels[2, 6:10, 6:10] = 121

    rgb1 = _labels_to_rgb(labels)
    rgb2 = _labels_to_rgb(labels)
    assert rgb1.dtype == np.uint8
    assert rgb1.shape == labels.shape + (3,)
    assert np.array_equal(rgb1, rgb2)
    assert not np.array_equal(rgb1[1, 4, 4], rgb1[2, 6, 6])

def test_link_workers_parallel_matches_single_thread():
    s0 = np.zeros((96, 96), dtype=np.int32)
    s1 = np.zeros((96, 96), dtype=np.int32)
    s2 = np.zeros((96, 96), dtype=np.int32)

    for idx, x in enumerate(range(8, 88, 12), start=1):
        s0[10:18, x:x + 8] = idx
        s1[11:19, x + 1:x + 9] = idx
        s2[12:20, x + 2:x + 10] = idx

    single = build_association_tracks([s0, s1, s2], link_iou=0.1, link_dist=20.0, max_gap=3, link_workers=1)
    parallel = build_association_tracks([s0, s1, s2], link_iou=0.1, link_dist=20.0, max_gap=3, link_workers=4)

    assert len(single) == len(parallel)
    sig_single = sorted((tuple(n.instance_id for n in tr.nodes), tuple(n.z for n in tr.nodes)) for tr in single)
    sig_parallel = sorted((tuple(n.instance_id for n in tr.nodes), tuple(n.z for n in tr.nodes)) for tr in parallel)
    assert sig_single == sig_parallel


def test_gpu_prefilter_path_preserves_linking_when_candidates_include_truth(monkeypatch):
    from cellpose.semi3d import linking

    s0 = np.zeros((64, 64), dtype=np.int32)
    s1 = np.zeros((64, 64), dtype=np.int32)
    s0[10:18, 10:18] = 1
    s0[30:38, 30:38] = 2
    s1[11:19, 11:19] = 1
    s1[31:39, 31:39] = 2

    base = linking.build_association_tracks([s0, s1], link_iou=0.1, link_dist=20.0, max_gap=2, gpu_prefilter=False)

    # force prefilter path with complete candidate coverage
    def fake_prefilter(active, current_nodes, link_dist, size_tolerance):
        return {i: list(range(len(current_nodes))) for i in range(len(active))}

    monkeypatch.setattr(linking, "_gpu_prefilter_candidates", fake_prefilter)
    with_prefilter = linking.build_association_tracks([s0, s1], link_iou=0.1, link_dist=20.0, max_gap=2, gpu_prefilter=True)

    sig_base = sorted((tuple(n.instance_id for n in tr.nodes), tuple(n.z for n in tr.nodes)) for tr in base)
    sig_pref = sorted((tuple(n.instance_id for n in tr.nodes), tuple(n.z for n in tr.nodes)) for tr in with_prefilter)
    assert sig_base == sig_pref
