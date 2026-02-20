import numpy as np

from cellpose.semi3d.linking import build_association_tracks
from cellpose.semi3d.refine import relabel_tracks
from cellpose.semi3d.training import dataset


def test_gap_tolerant_linking_and_mandatory_reconstruction():
    s0 = np.zeros((32, 32), dtype=np.int32)
    s1 = np.zeros((32, 32), dtype=np.int32)
    s2 = np.zeros((32, 32), dtype=np.int32)
    s0[10:14, 10:14] = 1
    s2[11:15, 11:15] = 1

    tracks = build_association_tracks([s0, s1, s2], link_iou=0.0, link_dist=10.0, max_gap=2)
    assert len(tracks) == 1

    img = np.stack([s0.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32)], axis=0)
    refined, _, reconstructed_flags, kept = relabel_tracks(tracks, img, min_track_len=1, min_conf=0.0)
    assert len(kept) == 1
    assert np.any(refined[1] > 0)
    assert np.any(reconstructed_flags[1] > 0)


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
