import csv
from pathlib import Path

from joplin_review.workflow import (
    DEFAULT_REVIEW,
    export_case_reviews,
    extract_case_id,
    extract_review_fields,
    import_case_reviews,
    render_note,
    review_record,
    stable_note_id,
)


class FakeJoplinClient:
    def __init__(self) -> None:
        self.notes = {}
        self.uploads = []

    def ensure_folder(self, title: str) -> str:
        return "folder-1"

    def get_note(self, note_id: str):
        return self.notes.get(note_id)

    def upload_resource(self, path: Path, title: str | None = None) -> str:
        self.uploads.append(path)
        return f"resource-{len(self.uploads)}"

    def upsert_note(self, note_id: str, title: str, body: str, parent_id: str):
        self.notes[note_id] = {
            "id": note_id,
            "title": title,
            "body": body,
            "parent_id": parent_id,
            "updated_time": 123,
        }
        return self.notes[note_id]

    def folder_notes(self, folder_id: str):
        return list(self.notes.values())


def test_review_note_round_trip():
    review = {**DEFAULT_REVIEW, "review_status": "REJECTED", "reviewer": "Domain reviewer"}
    body = render_note(
        "TOR10",
        {"status": "Rejected", "model": "Prithvi", "nws_available": "True"},
        review,
        "resource-1",
        None,
        Path("/tmp/TOR10/final_path_map.png"),
    )
    assert extract_case_id(body) == "TOR10"
    assert extract_review_fields(body)["review_status"] == "REJECTED"
    assert extract_review_fields(body)["reviewer"] == "Domain reviewer"


def test_label_ready_requires_real_geospatial_correction(tmp_path: Path):
    correction = tmp_path / "TOR10_corrected.geojson"
    correction.write_text('{"type":"FeatureCollection","features":[]}')
    body = render_note(
        "TOR10",
        {"status": "Rejected"},
        {**DEFAULT_REVIEW, "review_status": "CORRECTED", "corrected_path_file": str(correction)},
        "resource-1",
        None,
        tmp_path / "map.png",
    )
    record = review_record({"id": "note", "body": body}, tmp_path)
    assert record is not None
    assert record["label_ready"] is True


def test_export_is_idempotent_and_preserves_human_review(tmp_path: Path):
    results = tmp_path / "outputs"
    (results / "reports").mkdir(parents=True)
    case_dir = results / "cases" / "TOR5"
    case_dir.mkdir(parents=True)
    (case_dir / "final_path_map.png").write_bytes(b"fake-png")
    with (results / "reports" / "current_dataset_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "model", "status", "evaluation_role"])
        writer.writeheader()
        writer.writerow({"case_id": "TOR5", "model": "Prithvi", "status": "Unverified", "evaluation_role": "test"})

    client = FakeJoplinClient()
    first = export_case_reviews(client, results)
    note_id = stable_note_id("TOR5")
    client.notes[note_id]["body"] = client.notes[note_id]["body"].replace(
        "Review status: UNREVIEWED", "Review status: REJECTED"
    )
    second = export_case_reviews(client, results)
    assert first["synced"] == second["synced"] == 1
    assert len(client.uploads) == 1
    assert extract_review_fields(client.notes[note_id]["body"])["review_status"] == "REJECTED"


def test_import_writes_review_csv(tmp_path: Path):
    results = tmp_path / "outputs"
    client = FakeJoplinClient()
    body = render_note(
        "TOR12",
        {"status": "Rejected"},
        {**DEFAULT_REVIEW, "review_status": "UNCERTAIN"},
        "resource-1",
        None,
        tmp_path / "map.png",
    )
    client.notes[stable_note_id("TOR12")] = {
        "id": stable_note_id("TOR12"),
        "body": body,
        "updated_time": 123,
    }
    summary = import_case_reviews(client, results, tmp_path)
    assert summary["reviews"] == 1
    with open(summary["output"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case_id"] == "TOR12"
    assert rows[0]["review_status"] == "UNCERTAIN"
