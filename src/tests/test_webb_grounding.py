from __future__ import annotations

import json
from pathlib import Path

import grounding.ingest as ingest_module
from eval.webb import evaluate_webb_benchmark
from grounding.ingest import diff_webb_snapshot, ingest_webb_handbook, webb_sync
from grounding.provider import WebbGroundingProvider
from grounding.store import WebbKnowledgeStore
from serve.types import ChatMessage


DEMO_SEED_PACK = "data/webb/seed_urls_demo.json"
SOURCE_POLICIES = "data/webb/source_policies.json"
HANDBOOK_PATH = "data/webb/mock/handbook.txt"


def test_webb_demo_fixtures_follow_structured_fixture_standard():
    payload = json.loads(Path(DEMO_SEED_PACK).read_text())
    assert payload["fixture_schema_version"] == 2

    local_entries = [entry for entry in payload["entries"] if str(entry.get("url", "")).startswith("data/webb/mock/")]
    assert local_entries
    for entry in local_entries:
        assert entry["source_kind"] == "fixture"
        if entry["url"].endswith(".html"):
            assert entry["fixture_format"] == "structured_html"

    for path in Path("data/webb/mock").glob("*.html"):
        text = path.read_text()
        assert "<pre>" not in text
        assert '<script type="application/json">' in text


def test_webb_sync_populates_snapshot_and_queries(tmp_path: Path):
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    result = webb_sync(
        dsn,
        seed_url_pack=DEMO_SEED_PACK,
        source_policy_path=SOURCE_POLICIES,
        handbook_url=HANDBOOK_PATH,
        allow_ocr_fallback=False,
        label="pytest-webb",
    )
    assert result["snapshot_id"]
    assert result["quality"]["handbook_citable"] is True
    assert result["site"]["course_versions_ingested"] == result["families"]["course_catalog"]["rows"]["course_versions"]
    assert result["site"]["faculty_profiles_ingested"] == result["families"]["faculty"]["rows"]["faculty_profiles"]

    store = WebbKnowledgeStore(dsn)
    provider = WebbGroundingProvider(store)
    course_result = provider.query("What is Honors Museum Research at Webb?", route="course_catalog")
    assert course_result.hits
    assert "museum" in f"{course_result.hits[0].title} {course_result.hits[0].content}".lower()

    course_diff_result = provider.query(
        "How did math and computer science change from 2025-26 to 2026-27 at Webb?",
        route="course_catalog",
    )
    assert not course_diff_result.hits

    handbook_result = provider.query("What does the handbook say about academic honesty?", route="handbook")
    assert handbook_result.hits
    content = handbook_result.hits[0].content.lower()
    assert "academic integrity" in content
    assert "generated from/by ai" in content

    student_life_result = provider.query("What is dorm life like at Webb?", route="student_life")
    assert student_life_result.hits
    assert "dorm life is central to the webb experience" in student_life_result.hits[0].content.lower()

    athletics_result = provider.query("Did Girls Tennis beat Woodcrest Christian in 2025-26?", route="athletics")
    assert athletics_result.hits
    assert "win 12-6" in athletics_result.hits[0].content.lower()


def test_diff_webb_snapshot_marks_changed_and_unchanged(tmp_path: Path):
    course_2025 = tmp_path / "course_catalog_2025_26.html"
    course_2025.write_text(Path("data/webb/mock/course_catalog_2025_26.html").read_text())
    course_2026 = tmp_path / "course_catalog_2026_27.html"
    course_2026.write_text(Path("data/webb/mock/mathematics_computer_science_2026_27.html").read_text())
    faculty = tmp_path / "faculty.html"
    faculty.write_text(Path("data/webb/mock/faculty.html").read_text())
    admissions = tmp_path / "admissions.html"
    admissions.write_text(Path("data/webb/mock/admissions.html").read_text())
    publications = tmp_path / "publications.html"
    publications.write_text(Path("data/webb/mock/publications.html").read_text())
    handbook = tmp_path / "handbook.txt"
    handbook.write_text(Path(HANDBOOK_PATH).read_text())
    seed_pack = tmp_path / "seed_urls_demo.json"
    seed_pack.write_text(
        json.dumps(
            {
                "entries": [
                    {"url": str(course_2025), "page_type": "course", "department": "Math & Computer Science", "school_year": "2025-26"},
                    {"url": str(course_2026), "page_type": "course", "department": "Mathematics & Computer Science", "school_year": "2026-27"},
                    {"url": str(faculty), "page_type": "faculty"},
                    {"url": str(admissions), "page_type": "admissions"},
                    {"url": str(publications), "page_type": "publication", "school_year": "2026-27"},
                ]
            }
        )
    )
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    first = webb_sync(
        dsn,
        seed_url_pack=str(seed_pack),
        source_policy_path=SOURCE_POLICIES,
        handbook_url=str(handbook),
        label="snapshot-a",
    )
    admissions.write_text(admissions.read_text().replace("We look forward to seeing you soon!", "We look forward to seeing you on campus soon!"))
    second = webb_sync(
        dsn,
        seed_url_pack=str(seed_pack),
        source_policy_path=SOURCE_POLICIES,
        handbook_url=str(handbook),
        label="snapshot-b",
    )
    diff = diff_webb_snapshot(dsn, first["snapshot_id"], second["snapshot_id"])
    assert str(admissions) in diff["changed_urls"]
    assert str(course_2025) in diff["unchanged_urls"]


def test_ingest_webb_handbook_uses_ocr_fallback_when_native_text_is_degraded(
    monkeypatch,
    tmp_path: Path,
):
    pdf_path = tmp_path / "handbook.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake handbook")

    monkeypatch.setattr(
        "grounding.ingest._extract_pdf_text_pages",
        lambda _pdf_bytes: [{"page_number": 1, "text": ".... ... ,,,"}],
    )
    monkeypatch.setattr(
        "grounding.ingest._extract_pdf_pages_via_ocr",
        lambda _pdf_bytes: [{"page_number": 1, "text": "Attendance Policy Students are expected to attend all classes."}],
    )

    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    result = ingest_webb_handbook(
        dsn,
        str(pdf_path),
        allow_ocr_fallback=True,
    )
    assert result["extraction_quality"] == "ocr"


def test_webb_provider_routes_across_domains(tmp_path: Path):
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    webb_sync(
        dsn,
        seed_url_pack=DEMO_SEED_PACK,
        source_policy_path=SOURCE_POLICIES,
        handbook_url=HANDBOOK_PATH,
        label="routing",
    )
    provider = WebbGroundingProvider(WebbKnowledgeStore(dsn))

    assert provider.route_messages([ChatMessage(role="user", content="What does the handbook say about phones?")]).route == "handbook_policy"
    assert provider.route_messages([ChatMessage(role="user", content="Who is the Dean of College Guidance?")]).route == "faculty"
    assert provider.route_messages([ChatMessage(role="user", content="How do I apply to Webb?")]).route == "admissions_general"
    assert provider.route_messages([ChatMessage(role="user", content="How did CS offerings change from 2025-26 to 2026-27?")]).route == "course_catalog"
    assert provider.route_messages([ChatMessage(role="user", content="What is dorm life like at Webb?")]).route == "student_life"
    assert provider.route_messages([ChatMessage(role="user", content="What does Webb say about the Alf Museum?")]).route == "museum_programs"
    assert provider.route_messages([ChatMessage(role="user", content="Did Girls Tennis win against Woodcrest Christian?")]).route == "athletics"


def test_webb_sync_can_refresh_one_family_and_carry_forward_others(tmp_path: Path):
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    first = webb_sync(
        dsn,
        seed_url_pack=DEMO_SEED_PACK,
        source_policy_path=SOURCE_POLICIES,
        handbook_url=HANDBOOK_PATH,
        label="full-sync",
    )
    second = webb_sync(
        dsn,
        seed_url_pack=DEMO_SEED_PACK,
        source_policy_path=SOURCE_POLICIES,
        handbook_url=HANDBOOK_PATH,
        label="faculty-only-sync",
        families=["faculty"],
    )
    assert "course_catalog" in second["carried_forward_families"]
    assert second["families"]["faculty"]["refreshed"] is True
    assert second["families"]["course_catalog"]["carried_forward"] is True

    store = WebbKnowledgeStore(dsn)
    provider = WebbGroundingProvider(store, snapshot_id=second["snapshot_id"])
    assert provider.query("What is Honors Museum Research at Webb?", route="course_catalog").hits


def test_webb_provider_exposes_top2_fanout_for_mixed_queries(tmp_path: Path):
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    webb_sync(
        dsn,
        seed_url_pack=DEMO_SEED_PACK,
        source_policy_path=SOURCE_POLICIES,
        handbook_url=HANDBOOK_PATH,
        label="fanout",
    )
    provider = WebbGroundingProvider(WebbKnowledgeStore(dsn), route_fanout_limit=2)
    decision = provider.route_messages(
        [
            ChatMessage(
                role="user",
                content="I'm applying to Webb and want to know about dorm life too.",
            )
        ]
    )
    assert decision.metadata["low_confidence"] is True
    assert len(decision.metadata["fanout_routes"]) == 2
    assert "admissions_general" in decision.metadata["fanout_routes"]
    assert "student_life" in decision.metadata["fanout_routes"]


def test_read_bytes_uses_browserish_headers_for_remote_fetch(monkeypatch):
    observed = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"ok"

    def _fake_urlopen(request, timeout=30):
        observed["url"] = request.full_url
        observed["headers"] = {key.lower(): value for key, value in request.header_items()}
        observed["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(ingest_module.urllib.request, "urlopen", _fake_urlopen)
    payload = ingest_module._read_bytes("https://example.com/webb")
    assert payload == b"ok"
    assert observed["url"] == "https://example.com/webb"
    assert "mozilla/5.0" in observed["headers"]["user-agent"].lower()
    assert observed["headers"]["accept-language"].startswith("en-US")
    assert observed["timeout"] == 30


def test_webb_eval_falls_back_to_latest_completed_snapshot_when_sync_fails(
    monkeypatch,
    tmp_path: Path,
):
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    webb_sync(
        dsn,
        seed_url_pack=DEMO_SEED_PACK,
        source_policy_path=SOURCE_POLICIES,
        handbook_url=HANDBOOK_PATH,
        label="baseline",
    )
    responses_path = tmp_path / "webb.responses"
    responses_path.write_text(
        json.dumps(
            {
                "domain": "faculty",
                "response": "I found the Dean of College Guidance on the faculty page. [source: faculty]",
                "expected_substrings": ["dean of college guidance"],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "eval.webb.webb_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )
    result = evaluate_webb_benchmark(
        str(responses_path),
        grounding_dsn=dsn,
        seed_url_pack=DEMO_SEED_PACK,
        handbook_url=HANDBOOK_PATH,
        source_policy_path=SOURCE_POLICIES,
        sync_on_start=True,
    )
    assert result.examples == 1
