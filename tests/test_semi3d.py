import numpy as np

from cellpose.semi3d.linking import build_association_tracks
from cellpose.semi3d.refine import relabel_tracks
from cellpose.semi3d.training import dataset
from cellpose.cli import get_arg_parser


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
        "--semi3d_memmap_stage2_inputs", "--semi3d_stage2_use_gpu", "--semi3d_link_gpu_prefilter", "--semi3d_merge_dist", "10", "--semi3d_recon_min_free_fraction", "0.3", "--semi3d_save_debug_tiff"
    ])
    assert args.semi3d_stage2 is True
    assert args.semi3d_fill_edges is True
    assert args.semi3d_memmap_stage2_inputs is True
    assert args.semi3d_stage2_use_gpu is True
    assert args.semi3d_link_gpu_prefilter is True
    assert abs(args.semi3d_merge_dist - 10.0) < 1e-6
    assert abs(args.semi3d_recon_min_free_fraction - 0.3) < 1e-6
    assert args.semi3d_save_debug_tiff is True


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
