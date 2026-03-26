from progress import build_progress_snapshot


def test_build_progress_snapshot_formats_estimated_completion():
    snapshot = build_progress_snapshot(30.0, (50, 100))

    assert snapshot.fraction_complete == 0.5
    assert snapshot.remaining_seconds == 30.0
    assert snapshot.summary == "50.0% · 00:00:30 elapsed · 00:00:30 left"


def test_build_progress_snapshot_handles_unknown_totals():
    snapshot = build_progress_snapshot(15.0)

    assert snapshot.fraction_complete is None
    assert snapshot.remaining_seconds is None
    assert snapshot.summary == "??.?% · 00:00:15 elapsed · --:--:-- left"
