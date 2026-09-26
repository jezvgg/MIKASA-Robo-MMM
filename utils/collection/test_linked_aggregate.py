import errno, os
import pytest

from utils.collection import linked_aggregate as m


def test_shared_episode_file_and_chunk_rotation():
    frame = {
        "videos/hand/chunk_index": [0, 0, 1, 1],
        "videos/hand/file_index": [0, 0, 0, 1],
        "from": [0.0, 1.0, 0.0, 2.0],
    }
    m.remap_pairs(
        frame, "videos/hand", {(0, 0): (3, 1), (1, 0): (4, 0), (1, 1): (4, 1)}
    )
    assert frame["videos/hand/chunk_index"] == [3, 3, 4, 4]
    assert frame["videos/hand/file_index"] == [1, 1, 0, 1]
    assert frame["from"] == [0.0, 1.0, 0.0, 2.0]
    assert [m.file_index(n, 2) for n in range(3)] == [(0, 0), (0, 1), (1, 0)]


def test_hardlink_survives_removing_original_and_preserves_bytes(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    video = src / "clip.mp4"
    video.write_bytes(b"original encoded bytes")
    dest = tmp_path / "out/clip.mp4"
    row = m.link_video(video, dest, src)
    m.verify_links([row])
    assert os.path.samefile(video, dest)
    video.unlink()
    assert dest.read_bytes() == b"original encoded bytes"


def test_existing_output_and_symlinks_are_never_overwritten(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    v = src / "clip.mp4"
    v.write_bytes(b"original")
    out = tmp_path / "dest.mp4"
    out.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        m.link_video(v, out, src)
    link = src / "symlink.mp4"
    link.symlink_to(v)
    with pytest.raises(ValueError):
        m.link_video(link, tmp_path / "new.mp4", src)
    assert out.read_bytes() == b"preserve" and v.read_bytes() == b"original"


def test_cross_filesystem_has_no_copy_fallback(tmp_path, monkeypatch):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"original")
    out = tmp_path / "out.mp4"

    def cross(*args, **kwargs):
        raise OSError(errno.EXDEV, "different filesystem")

    monkeypatch.setattr(m.os, "link", cross)
    with pytest.raises(OSError) as error:
        m.link_video(v, out, tmp_path)
    assert error.value.errno == errno.EXDEV and not out.exists()


def test_detects_source_modification(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"original")
    out = tmp_path / "out.mp4"
    row = m.link_video(v, out, tmp_path)
    v.write_bytes(b"changed bytes")
    with pytest.raises(ValueError):
        m.verify_links([row])
