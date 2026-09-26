"""Encode compact RGB only when every frame passes the render-quality budget."""

import hashlib
from pathlib import Path

import numpy as np


PROFILES = {
    "reference": {"crfs": (12,), "gop": 2},
    "compact": {"crfs": (16, 12, 0), "gop": 12},
}


def video_settings(profile):
    """Keep the requested policy separate from each camera's actual encoding."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown video profile: {profile}")
    settings = PROFILES[profile]
    return dict(
        codec="h264",
        profile=profile,
        crf_candidates=list(settings["crfs"]),
        pixel_format="yuv444p",
        decoded_format="RGB",
        gop=settings["gop"],
        max_mean_error_255=2.0,
        actual_settings="episodes[].video_encoding per camera",
    )


def measure_encoded_video(images, video):
    """Compare every decoded frame with its exact lossless PNG source."""
    import av
    from PIL import Image

    paths = sorted(Path(images).glob("frame-[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not paths:
        raise ValueError("No source video frames")
    error_sum = pixels = count = 0
    digest = hashlib.sha256()
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(paths):
                raise ValueError("Encoded video has extra frames")
            actual = frame.to_ndarray(format="rgb24")
            with Image.open(paths[index]) as image:
                expected = np.asarray(image.convert("RGB"))
            if actual.shape != expected.shape:
                raise ValueError("Encoded video changed the RGB dimensions")
            error_sum += int(
                np.abs(actual.astype(np.int16) - expected.astype(np.int16)).sum()
            )
            pixels += actual.size
            digest.update(actual.tobytes())
            count += 1
    if count != len(paths):
        raise ValueError("Encoded video is missing source frames")
    return dict(
        frames=count,
        mean_absolute_error_255=error_sum / pixels,
        decoded_rgb_sha256=digest.hexdigest(),
    )


def encode_checked_video(images, target, fps, *, crfs=(16, 12, 0), gop=12):
    """Try stronger compression first; retain source images on any failure.

    Malformed output fails immediately. Only excessive pixel error triggers a
    lower CRF, ending at lossless codec quantization (RGB/YUV conversion remains).
    """
    from lerobot.datasets.video_utils import encode_video_frames

    target = Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if (
        not crfs
        or any(type(c) is not int or not 0 <= c <= 51 for c in crfs)
        or any(a <= b for a, b in zip(crfs, crfs[1:]))
        or type(gop) is not int
        or gop < 1
    ):
        raise ValueError("Use decreasing CRFs in [0,51] and a positive GOP")
    trials = []
    for crf in crfs:
        encode_video_frames(
            images,
            target,
            fps,
            vcodec="h264",
            crf=crf,
            pix_fmt="yuv444p",
            g=gop,
            overwrite=False,
        )
        measured = measure_encoded_video(images, target)
        trials.append(
            dict(
                crf=crf,
                bytes=target.stat().st_size,
                mean_absolute_error_255=measured["mean_absolute_error_255"],
            )
        )
        if measured["mean_absolute_error_255"] <= 2.0:
            return dict(
                codec="h264",
                crf=crf,
                gop=gop,
                pixel_format="yuv444p",
                trials=trials,
                **measured,
            )
        target.unlink()  # Only this invocation's rejected temporary output.
    raise ValueError("No encoding satisfied checklist E4; source frames retained")
