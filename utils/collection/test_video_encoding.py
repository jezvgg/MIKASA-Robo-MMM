"""Real codec failures must not silently lose frames or exceed the E4 limit."""

import numpy as np
import pytest

from utils.collection.video_encoding import encode_checked_video, measure_encoded_video


def write_frames(root, count):
    from PIL import Image

    root.mkdir()
    rng = np.random.default_rng(18)
    for index in range(count):
        # High-frequency, non-square images exercise quality and shape handling.
        pixels = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"frame-{index:06d}.png")


def test_real_codec_falls_back_without_losing_source_images(tmp_path):
    pytest.importorskip("lerobot")
    images, target = tmp_path / "images", tmp_path / "video.mp4"
    write_frames(images, 4)
    result = encode_checked_video(images, target, 10, crfs=(40, 0), gop=12)
    assert result["crf"] == 0
    assert result["trials"][0]["mean_absolute_error_255"] > 2
    assert result["mean_absolute_error_255"] <= 2
    assert result["frames"] == 4 and len(list(images.glob("*.png"))) == 4
    assert (
        measure_encoded_video(images, target)["decoded_rgb_sha256"]
        == result["decoded_rgb_sha256"]
    )
    previous = target.read_bytes()
    with pytest.raises(FileExistsError):
        encode_checked_video(images, target, 10)
    assert target.read_bytes() == previous


def test_real_decoder_detects_missing_frames(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.video_utils import encode_video_frames

    images, target = tmp_path / "images", tmp_path / "video.mp4"
    write_frames(images, 3)
    last = images / "frame-000002.png"
    hidden = images / "saved.png"
    last.rename(hidden)
    encode_video_frames(images, target, 10, vcodec="h264", pix_fmt="yuv444p", crf=0)
    hidden.rename(last)
    with pytest.raises(ValueError, match="missing source frames"):
        measure_encoded_video(images, target)
    assert len(list(images.glob("frame-*.png"))) == 3


def test_lerobot_reads_concatenated_episodes_with_different_fallback_crfs(
    tmp_path, monkeypatch
):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from utils.collection.contract import CAMERAS
    from utils.collection.export_lerobot import dataset_class
    from utils.collection.video_encoding import PROFILES

    # Force both encoding branches using real pixels, then test actual LeRobot
    # seeking across the boundary of two concatenated episode bitstreams.
    monkeypatch.setitem(PROFILES, "compact", {"crfs": (40, 0), "gop": 12})
    features = {
        "action": {"dtype": "float32", "shape": (13,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (12,), "names": None},
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
            for camera, shape in CAMERAS.items()
        },
    }
    root = tmp_path / "dataset"
    writer = dataset_class("compact").create(
        repo_id="local/codec-test",
        root=root,
        fps=10,
        robot_type="ds_fetch",
        features=features,
        use_videos=True,
        image_writer_threads=1,
        video_backend="pyav",
        vcodec="h264",
    )
    originals = []
    rng = np.random.default_rng(35)
    try:
        for episode in range(2):
            for _ in range(4):
                images = {
                    camera: (
                        np.zeros(shape, dtype=np.uint8)
                        if episode == 0
                        else rng.integers(0, 256, shape, dtype=np.uint8)
                    )
                    for camera, shape in CAMERAS.items()
                }
                originals.append(images)
                writer.add_frame(
                    {
                        "action": np.zeros(13, dtype=np.float32),
                        "observation.state": np.zeros(12, dtype=np.float32),
                        "task": "Test video episode boundaries.",
                        **{f"observation.images.{k}": v for k, v in images.items()},
                    }
                )
            writer.save_episode(parallel_encoding=False)
        encodings = writer._mikasa_video_encoding
    finally:
        writer.finalize()
        writer.stop_image_writer()
    assert {v["crf"] for v in encodings[0].values()} == {40}
    assert {v["crf"] for v in encodings[1].values()} == {0}
    reader = LeRobotDataset("local/codec-test", root=root, video_backend="pyav")
    assert len(reader) == 8
    for index in reversed(range(8)):
        sample = reader[index]
        for camera in CAMERAS:
            actual = np.rint(
                sample[f"observation.images.{camera}"].numpy().transpose(1, 2, 0) * 255
            ).astype(np.int16)
            assert (
                np.abs(actual - originals[index][camera].astype(np.int16)).mean() <= 2
            )
