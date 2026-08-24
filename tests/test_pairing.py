from pathlib import Path

from src.pairing import find_image_pairs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_pairing_accepts_common_before_after_patterns(tmp_path):
    _touch(tmp_path / "TOR1_before.tif")
    _touch(tmp_path / "TOR1_after.tif")
    _touch(tmp_path / "TOR2_pre_event.tif")
    _touch(tmp_path / "TOR2_post_event.tif")
    _touch(tmp_path / "before_TOR3.tif")
    _touch(tmp_path / "after_TOR3.tif")

    df = find_image_pairs(tmp_path)

    ok = df[df["status"].eq("OK")]
    assert set(ok["tornado_id"]) == {"TOR1", "TOR2", "TOR3"}
    assert len(ok) == 3


def test_pairing_reports_ambiguous_candidates(tmp_path):
    _touch(tmp_path / "TOR7_before.tif")
    _touch(tmp_path / "TOR7_before_copy.tif")
    _touch(tmp_path / "TOR7_after.tif")

    df = find_image_pairs(tmp_path)

    row = df[df["pair_key"].eq("TOR7")].iloc[0]
    assert "AMBIGUOUS_BEFORE" in row["status"]
    assert row["before_path"] == ""
    assert "multiple BEFORE" in row["reason"]
