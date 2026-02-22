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
        "--semi3d_memmap_stage2_inputs", "--semi3d_stage2_use_gpu", "--semi3d_link_gpu_prefilter", "--semi3d_merge_dist", "10", "--semi3d_recon_min_free_fraction", "0.3"
    ])
    assert args.semi3d_stage2 is True
    assert args.semi3d_fill_edges is True
    assert args.semi3d_memmap_stage2_inputs is True
    assert args.semi3d_stage2_use_gpu is True
    assert args.semi3d_link_gpu_prefilter is True
    assert abs(args.semi3d_merge_dist - 10.0) < 1e-6
    assert abs(args.semi3d_recon_min_free_fraction - 0.3) < 1e-6
