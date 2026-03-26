from __future__ import annotations

from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any

from grounding.store import CatalogStore, WebbKnowledgeStore


SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"<h([1-3])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,8}\s?-?\d{2,4}[A-Z]?)\b")
SCHOOL_YEAR_RE = re.compile(r"(20\d{2}\s*[-–]\s*\d{2})")

DEFAULT_FAMILY_FRESHNESS = {
    "athletics": {"active_season": "6h", "off_season": "24h"},
    "faculty": {"cadence": "24h"},
    "admissions_general": {"cadence": "24h"},
    "student_life": {"cadence": "24h"},
    "college_guidance": {"cadence": "24h"},
    "course_catalog": {"cadence": "168h"},
    "handbook_policy": {"cadence": "168h"},
    "academic_publications": {"cadence": "168h"},
    "mission_values": {"cadence": "168h"},
    "museum_programs": {"cadence": "168h"},
}

PAGE_TYPE_TO_FAMILY = {
    "course": "course_catalog",
    "faculty": "faculty",
    "handbook": "handbook_policy",
    "handbook_policy": "handbook_policy",
    "admissions": "admissions_general",
    "admissions_general": "admissions_general",
    "mission_values": "mission_values",
    "college_guidance": "college_guidance",
    "student_life": "student_life",
    "culture_community": "student_life",
    "museum_programs": "museum_programs",
    "unique_programs": "museum_programs",
    "athletics": "athletics",
    "publication": "academic_publications",
}

ALL_WEBB_FAMILIES = sorted(DEFAULT_FAMILY_FRESHNESS.keys())
DEFAULT_WEBB_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def ingest_catalog(dsn: str, input_path: str) -> None:
    store = CatalogStore(dsn)
    store.create_schema()
    payload = json.loads(Path(input_path).read_text())
    if "institutions" in payload:
        store.upsert_institutions(payload["institutions"])
    if "terms" in payload:
        store.upsert_terms(payload["terms"])
    if "programs" in payload:
        store.upsert_programs(payload["programs"])
    if "courses" in payload:
        store.upsert_courses(payload["courses"])
    if "sections" in payload:
        store.upsert_sections(payload["sections"])


def _stable_id(prefix: str, payload: Any) -> str:
    digest = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"{prefix}-{__import__('hashlib').sha256(digest).hexdigest()[:24]}"


def _normalize_space(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split()).strip()


def _normalize_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _normalize_space(value).lower()).strip("-")


def _looks_like_url(path_or_url: str) -> bool:
    return "://" in path_or_url and not Path(path_or_url).exists()


def _read_bytes(path_or_url: str) -> bytes:
    local = Path(path_or_url)
    if local.exists():
        return local.read_bytes()
    if path_or_url.startswith("file://"):
        return Path(path_or_url.removeprefix("file://")).read_bytes()
    request = urllib.request.Request(path_or_url, headers=DEFAULT_WEBB_HTTP_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch Webb source {path_or_url!r}: HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch Webb source {path_or_url!r}: {exc.reason}.") from exc


def _read_text(path_or_url: str) -> str:
    raw = _read_bytes(path_or_url)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _load_seed_entries(seed_url_pack: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(seed_url_pack).read_text())
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("seed_urls") or payload.get("sources") or []
        return [entry for entry in entries if isinstance(entry, dict)]
    raise ValueError(f"Unsupported seed pack format in {seed_url_pack!r}")


def _load_source_policies(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    target = Path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text())
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        rules = payload.get("rules") or payload.get("page_type_rules") or []
        return [entry for entry in rules if isinstance(entry, dict)]
    return []


def _extract_title(html_text: str) -> str:
    title_match = TITLE_RE.search(html_text)
    if title_match:
        return _normalize_space(html.unescape(TAG_RE.sub(" ", title_match.group(1))))
    heading_match = HEADING_RE.search(html_text)
    if heading_match:
        return _normalize_space(html.unescape(TAG_RE.sub(" ", heading_match.group(2))))
    return "Untitled Webb source"


def _extract_headings(html_text: str) -> list[str]:
    return [
        _normalize_space(html.unescape(TAG_RE.sub(" ", match.group(2))))
        for match in HEADING_RE.finditer(html_text)
        if _normalize_space(html.unescape(TAG_RE.sub(" ", match.group(2))))
    ]


def _extract_main_html(html_text: str) -> str:
    for pattern in (r"<main[^>]*>(.*?)</main>", r"<article[^>]*>(.*?)</article>", r"<body[^>]*>(.*?)</body>"):
        match = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    return html_text


def _html_to_text(html_text: str) -> str:
    focused = _extract_main_html(html_text)
    without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", focused, flags=re.IGNORECASE | re.DOTALL)
    with_breaks = re.sub(r"</(p|div|section|article|li|h1|h2|h3|tr)>", "\n", without_scripts, flags=re.IGNORECASE)
    stripped = TAG_RE.sub(" ", with_breaks)
    unescaped = html.unescape(stripped)
    lines = [_normalize_space(line) for line in unescaped.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _iter_json_objects(payload: Any):
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_json_objects(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_json_objects(item)


def _extract_script_payloads(html_text: str) -> list[Any]:
    payloads: list[Any] = []
    for match in SCRIPT_RE.finditer(html_text):
        content = match.group(1).strip()
        if not content or "{" not in content and "[" not in content:
            continue
        for candidate in (content, html.unescape(content)):
            try:
                payloads.append(json.loads(candidate))
                break
            except json.JSONDecodeError:
                continue
    return payloads


def _infer_page_type(source_url: str, page_title: str, policies: list[dict[str, Any]], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    haystack = f"{source_url} {page_title}".lower()
    for policy in policies:
        needle = str(policy.get("contains", "")).lower().strip()
        if needle and needle in haystack:
            return str(policy.get("page_type", "generic"))
    if "curriculum-detail" in source_url or "course-catalog" in source_url:
        return "course"
    if "faculty" in source_url or "directory" in source_url:
        return "faculty"
    if "admission" in source_url or "how-to-apply" in source_url or "tuition" in source_url:
        return "admissions"
    if "student-life" in source_url or "culture-and-community" in source_url:
        return "student_life"
    if "college-guidance" in source_url:
        return "college_guidance"
    if "mission" in source_url or "values" in source_url:
        return "mission_values"
    if "museum" in source_url or "unique-learning" in source_url:
        return "museum_programs"
    if "athletics" in source_url or "athletic-teams" in source_url:
        return "athletics"
    if "publication" in source_url:
        return "publication"
    return "generic"


def _family_for_page_type(page_type: str) -> str:
    return PAGE_TYPE_TO_FAMILY.get(page_type, page_type)


def _normalize_route_family(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _family_for_page_type(_normalize_space(value))
    aliases = {
        "handbook": "handbook_policy",
        "admissions": "admissions_general",
        "publication": "academic_publications",
        "college_guidance": "college_guidance",
        "mission_values": "mission_values",
        "student_life": "student_life",
        "athletics": "athletics",
    }
    return aliases.get(normalized, normalized)


def _normalize_selected_families(families: list[str] | None) -> list[str]:
    if not families:
        return list(ALL_WEBB_FAMILIES)
    normalized = []
    seen: set[str] = set()
    for family in families:
        normalized_family = _normalize_route_family(family)
        if normalized_family is None or normalized_family not in ALL_WEBB_FAMILIES or normalized_family in seen:
            continue
        seen.add(normalized_family)
        normalized.append(normalized_family)
    return normalized or list(ALL_WEBB_FAMILIES)


def _family_metadata_template(families: list[str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for family in families:
        metadata[family] = {
            "refreshed": False,
            "carried_forward": False,
            "documents": 0,
            "chunks": 0,
            "rows": {},
            "freshness_policy": DEFAULT_FAMILY_FRESHNESS.get(family, {}),
        }
    return metadata


def _record_family_metric(
    family_metadata: dict[str, dict[str, Any]],
    family: str,
    key: str,
    amount: int = 1,
) -> None:
    entry = family_metadata.setdefault(
        family,
        {
            "refreshed": False,
            "carried_forward": False,
            "documents": 0,
            "chunks": 0,
            "rows": {},
            "freshness_policy": DEFAULT_FAMILY_FRESHNESS.get(family, {}),
        },
    )
    entry["refreshed"] = True
    if key in {"documents", "chunks"}:
        entry[key] = int(entry.get(key, 0)) + int(amount)
        return
    rows = dict(entry.get("rows") or {})
    rows[key] = int(rows.get(key, 0)) + int(amount)
    entry["rows"] = rows


def _school_year(entry: dict[str, Any], page_title: str, text: str) -> str | None:
    for source in (entry.get("school_year"), page_title, text):
        match = SCHOOL_YEAR_RE.search(str(source or ""))
        if match:
            return match.group(1).replace("–", "-").replace(" ", "")
    return None


def _chunk_text(text: str, *, heading: str | None, size: int = 500) -> list[str]:
    paragraphs = [_normalize_space(part) for part in re.split(r"\n{2,}|\n", text) if _normalize_space(part)]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n{paragraph}".strip()
        if buffer and len(candidate) > size:
            chunks.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    if not chunks and text.strip():
        chunks.append(_normalize_space(text))
    if heading:
        return [f"{heading}\n{chunk}".strip() for chunk in chunks]
    return chunks


def _course_rows_from_json(
    payloads: list[Any],
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
    school_year: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for item in _iter_json_objects(payload):
            title = _normalize_space(item.get("title") or item.get("courseTitle") or item.get("name"))
            description = _normalize_space(item.get("description") or item.get("courseDescription"))
            code = _normalize_space(item.get("code") or item.get("courseCode"))
            department = _normalize_space(item.get("department") or item.get("departmentName") or entry.get("department"))
            if not title or not description or not (code or department):
                continue
            course_key = _normalize_key(code or title)
            teachers = item.get("teachers") or item.get("faculty") or item.get("instructors") or []
            teacher_names = teachers if isinstance(teachers, list) else [_normalize_space(_field) for _field in [teachers] if _normalize_space(_field)]
            rows.append(
                {
                    "id": _stable_id("course", [source_url, school_year, course_key, description]),
                    "snapshot_id": entry["_snapshot_id"],
                    "course_key": course_key,
                    "department": department or None,
                    "course_code": code or None,
                    "title": title,
                    "school_year": school_year,
                    "level_marker": _normalize_space(item.get("level") or item.get("levelMarker")) or None,
                    "grade_eligibility": _normalize_space(item.get("grades") or item.get("gradeEligibility")) or None,
                    "prerequisites": _normalize_space(item.get("prerequisites") or item.get("prereqs")) or None,
                    "description": description,
                    "teacher_names": {"names": teacher_names} if teacher_names else None,
                    "source_url": source_url,
                    "citation_label": code or title,
                    "content_hash": _stable_id("hash", [title, code, description, school_year]),
                    "extra": {"page_title": page_title},
                    "current": False,
                }
            )
    return rows


def _course_rows_from_text(
    text: str,
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
    school_year: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    for block in blocks:
        lines = [_normalize_space(line) for line in block.splitlines() if _normalize_space(line)]
        if not lines:
            continue
        heading = lines[0]
        code_match = COURSE_CODE_RE.search(heading)
        if code_match is None:
            continue
        code = _normalize_space(code_match.group(1))
        title = _normalize_space(heading.replace(code_match.group(1), "").strip(" :-\u2014")) or page_title
        description = _normalize_space(" ".join(lines[1:])) or None
        rows.append(
            {
                "id": _stable_id("course", [source_url, school_year, code, description]),
                "snapshot_id": entry["_snapshot_id"],
                "course_key": _normalize_key(code or title),
                "department": _normalize_space(entry.get("department")) or None,
                "course_code": code or None,
                "title": title,
                "school_year": school_year,
                "level_marker": None,
                "grade_eligibility": None,
                "prerequisites": None,
                "description": description,
                "teacher_names": None,
                "source_url": source_url,
                "citation_label": code or title,
                "content_hash": _stable_id("hash", [title, code, description, school_year]),
                "extra": {"page_title": page_title, "parser": "text_fallback"},
                "current": False,
            }
        )
    if rows:
        return rows

    lines = [_normalize_space(line) for line in text.splitlines() if _normalize_space(line)]
    department = _normalize_space(entry.get("department")) or None
    stop_markers = {"meet our faculty"}

    def _looks_like_course_title(line: str) -> bool:
        lowered = line.lower()
        if lowered in stop_markers:
            return False
        if department and lowered == department.lower():
            return False
        if lowered == _normalize_space(page_title).lower():
            return False
        if "@" in line or lowered.startswith("photo of "):
            return False
        if lowered.startswith(("prerequisite", "prerequisites", "note:", "list of ")):
            return False
        if lowered.startswith(
            (
                "this course ",
                "in this course",
                "students ",
                "taught in ",
                "all freshmen ",
                "health & living ",
                "advanced courses in computer science are offered",
                "honors sinfonia is open",
                "precalculus ensures ",
                "the world language department ",
                "from the ",
                "what is ",
                "why do ",
                "are humans ",
                "for at least ",
                "many religious traditions ",
                "as nelson mandela said",
            )
        ):
            return False
        if SCHOOL_YEAR_RE.search(line):
            return False
        if re.match(r"^\d", line):
            return False
        if line.endswith("."):
            return False
        if len(line.split()) > 14:
            return False
        return True

    index = 0
    while index < len(lines):
        line = lines[index]
        lowered = line.lower()
        if lowered in stop_markers:
            break
        if not _looks_like_course_title(line):
            index += 1
            continue
        title = line
        index += 1
        description_lines: list[str] = []
        prerequisites: str | None = None
        while index < len(lines):
            current = lines[index]
            lowered_current = current.lower()
            if lowered_current in stop_markers:
                break
            if _looks_like_course_title(current):
                break
            if lowered_current.startswith("prerequisite") or lowered_current.startswith("prerequisites"):
                prerequisite_text = current.split(":", 1)[1] if ":" in current else current
                prerequisites = _normalize_space(prerequisite_text)
            else:
                description_lines.append(current)
            index += 1
        description = _normalize_space(" ".join(description_lines)) or None
        if not description and not prerequisites:
            continue
        rows.append(
            {
                "id": _stable_id("course", [source_url, school_year, title, description, prerequisites]),
                "snapshot_id": entry["_snapshot_id"],
                "course_key": _normalize_key(title),
                "department": department,
                "course_code": None,
                "title": title,
                "school_year": school_year,
                "level_marker": None,
                "grade_eligibility": None,
                "prerequisites": prerequisites,
                "description": description,
                "teacher_names": None,
                "source_url": source_url,
                "citation_label": title,
                "content_hash": _stable_id("hash", [title, description, prerequisites, school_year]),
                "extra": {"page_title": page_title, "parser": "title_block_fallback"},
                "current": False,
            }
        )
    return rows


def _faculty_rows_from_payloads(
    payloads: list[Any],
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for item in _iter_json_objects(payload):
            name = _normalize_space(item.get("name") or item.get("fullName"))
            role_title = _normalize_space(item.get("role") or item.get("title"))
            department = _normalize_space(item.get("department"))
            bio = _normalize_space(item.get("bio") or item.get("description"))
            if not name or not (role_title or department or bio):
                continue
            faculty_key = _normalize_key(name)
            rows.append(
                {
                    "id": _stable_id("faculty", [source_url, faculty_key, role_title, bio]),
                    "snapshot_id": entry["_snapshot_id"],
                    "faculty_key": faculty_key,
                    "name": name,
                    "role_title": role_title or None,
                    "department": department or None,
                    "bio": bio or None,
                    "source_url": source_url,
                    "citation_label": name,
                    "content_hash": _stable_id("hash", [name, role_title, department, bio]),
                    "extra": {"page_title": page_title},
                    "current": False,
                }
            )
    return rows


def _faculty_rows_from_text(
    text: str,
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    for block in blocks:
        lines = [_normalize_space(line) for line in block.splitlines() if _normalize_space(line)]
        if len(lines) < 2:
            continue
        name = lines[0]
        role_title = lines[1]
        bio = _normalize_space(" ".join(lines[2:])) or None
        if len(name.split()) < 2:
            continue
        rows.append(
            {
                "id": _stable_id("faculty", [source_url, name, role_title, bio]),
                "snapshot_id": entry["_snapshot_id"],
                "faculty_key": _normalize_key(name),
                "name": name,
                "role_title": role_title,
                "department": _normalize_space(entry.get("department")) or None,
                "bio": bio,
                "source_url": source_url,
                "citation_label": name,
                "content_hash": _stable_id("hash", [name, role_title, bio]),
                "extra": {"page_title": page_title, "parser": "text_fallback"},
                "current": False,
            }
        )
    return rows


def _admissions_rows_from_text(
    text: str,
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(_chunk_text(text, heading=page_title)):
        rows.append(
            {
                "id": _stable_id("admissions", [source_url, index, chunk]),
                "snapshot_id": entry["_snapshot_id"],
                "fact_key": _normalize_key(f"{page_title}-{index}"),
                "title": page_title,
                "text": chunk,
                "source_url": source_url,
                "citation_label": page_title,
                "content_hash": _stable_id("hash", [page_title, chunk]),
                "extra": {"page_title": page_title},
                "current": False,
            }
        )
    return rows


def _publication_rows_from_text(
    text: str,
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
    school_year: str | None,
) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    return [
        {
            "id": _stable_id("publication", [source_url, school_year, text[:240]]),
            "snapshot_id": entry["_snapshot_id"],
            "publication_key": _normalize_key(page_title),
            "title": page_title,
            "school_year": school_year,
            "text": text,
            "source_url": source_url,
            "citation_label": page_title,
            "content_hash": _stable_id("hash", [page_title, school_year, text]),
            "extra": {"page_title": page_title},
            "current": False,
        }
    ]


def _athletics_rows_from_payloads(
    payloads: list[Any],
    *,
    entry: dict[str, Any],
    page_title: str,
    source_url: str,
    school_year: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    teams: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for payload in payloads:
        team_entries = []
        if isinstance(payload, dict):
            if isinstance(payload.get("teams"), list):
                team_entries.extend([item for item in payload.get("teams", []) if isinstance(item, dict)])
            if isinstance(payload.get("athletics"), dict) and isinstance(payload["athletics"].get("teams"), list):
                team_entries.extend(
                    [item for item in payload["athletics"].get("teams", []) if isinstance(item, dict)]
                )
        for item in team_entries:
            team_name = _normalize_space(item.get("team_name") or item.get("team") or item.get("name"))
            if not team_name:
                continue
            season = _normalize_space(item.get("season") or school_year) or None
            team_key = _normalize_key(team_name)
            summary = _normalize_space(item.get("summary") or item.get("description")) or None
            coach_name = _normalize_space(item.get("coach") or item.get("head_coach")) or None
            teams.append(
                {
                    "id": _stable_id("ath-team", [source_url, team_key, season, coach_name]),
                    "snapshot_id": entry["_snapshot_id"],
                    "team_key": team_key,
                    "season": season,
                    "team_name": team_name,
                    "coach_name": coach_name,
                    "summary": summary,
                    "source_url": source_url,
                    "citation_label": team_name,
                    "content_hash": _stable_id("hash", [team_name, season, coach_name, summary]),
                    "extra": {"page_title": page_title},
                    "current": False,
                }
            )
            for index, game in enumerate(item.get("games") or []):
                if not isinstance(game, dict):
                    continue
                opponent = _normalize_space(game.get("opponent"))
                if not opponent:
                    continue
                game_date = _normalize_space(game.get("date") or game.get("game_date")) or None
                result = _normalize_space(game.get("result")) or None
                venue = _normalize_space(game.get("venue") or game.get("home_away")) or None
                games.append(
                    {
                        "id": _stable_id("ath-game", [source_url, team_key, season, index, opponent, game_date, result]),
                        "snapshot_id": entry["_snapshot_id"],
                        "game_key": _normalize_key(f"{team_name}-{season or 'current'}-{game_date or index}-{opponent}"),
                        "team_key": team_key,
                        "season": season,
                        "team_name": team_name,
                        "opponent": opponent,
                        "game_date": game_date,
                        "result": result,
                        "venue": venue,
                        "source_url": source_url,
                        "citation_label": team_name,
                        "content_hash": _stable_id("hash", [team_name, season, opponent, game_date, result, venue]),
                        "extra": {"page_title": page_title},
                        "current": False,
                    }
                )
            for index, record in enumerate(item.get("records") or []):
                if not isinstance(record, dict):
                    continue
                record_type = _normalize_space(record.get("type") or record.get("record_type"))
                value = _normalize_space(record.get("value"))
                if not record_type or not value:
                    continue
                records.append(
                    {
                        "id": _stable_id("ath-record", [source_url, team_key, season, index, record_type, value]),
                        "snapshot_id": entry["_snapshot_id"],
                        "record_key": _normalize_key(f"{team_name}-{record_type}"),
                        "season": season,
                        "team_name": team_name,
                        "record_type": record_type,
                        "value": value,
                        "source_url": source_url,
                        "citation_label": team_name,
                        "content_hash": _stable_id("hash", [team_name, season, record_type, value]),
                        "extra": {"page_title": page_title},
                        "current": False,
                    }
                )
    return teams, games, records


def _build_document_row(
    *,
    entry: dict[str, Any],
    source_url: str,
    page_type: str,
    page_title: str,
    text: str,
    school_year: str | None,
    raw_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    document_key = _normalize_key(source_url)
    source_family = _family_for_page_type(page_type)
    return {
        "id": _stable_id("document", [entry["_snapshot_id"], source_url]),
        "snapshot_id": entry["_snapshot_id"],
        "document_key": document_key,
        "source_family": source_family,
        "source_type": page_type,
        "page_type": page_type,
        "page_title": page_title,
        "source_url": source_url,
        "school_year": school_year,
        "citation_label": page_title,
        "content_hash": _stable_id("hash", [source_url, text]),
        "text": text,
        "raw_payload": raw_payload,
        "extra": {"entry": {key: value for key, value in entry.items() if not key.startswith("_")}},
        "current": False,
    }


def _build_chunk_rows(document_row: dict[str, Any], headings: list[str]) -> list[dict[str, Any]]:
    heading = headings[0] if headings else document_row["page_title"]
    chunks = _chunk_text(document_row["text"], heading=heading)
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        rows.append(
            {
                "id": _stable_id("chunk", [document_row["id"], index, chunk]),
                "snapshot_id": document_row["snapshot_id"],
                "document_id": document_row["id"],
                "source_family": document_row["source_family"],
                "source_type": document_row["source_type"],
                "school_year": document_row["school_year"],
                "heading": heading,
                "chunk_index": index,
                "citation_label": document_row["citation_label"],
                "content_hash": _stable_id("hash", [document_row["source_url"], index, chunk]),
                "text": chunk,
                "normalized_text": _normalize_space(chunk).lower(),
                "extra": {"page_title": document_row["page_title"]},
                "current": False,
            }
        )
    return rows


def ingest_webb_site(
    dsn: str,
    seed_url_pack: str,
    *,
    source_policy_path: str | None = None,
    snapshot_id: str | None = None,
    label: str = "webb-site-sync",
    families: list[str] | None = None,
    complete_snapshot: bool = True,
) -> dict[str, Any]:
    store = WebbKnowledgeStore(dsn)
    store.create_schema()
    selected_families = _normalize_selected_families(families)
    active_snapshot_id = snapshot_id or store.create_snapshot(
        label=label,
        seed_url_pack=seed_url_pack,
        metadata={"created_at": datetime.now(timezone.utc).isoformat()},
    )
    policies = _load_source_policies(source_policy_path)
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    courses: list[dict[str, Any]] = []
    faculty: list[dict[str, Any]] = []
    admissions: list[dict[str, Any]] = []
    publications: list[dict[str, Any]] = []
    athletics_teams: list[dict[str, Any]] = []
    athletics_games: list[dict[str, Any]] = []
    athletics_records: list[dict[str, Any]] = []
    family_metadata = _family_metadata_template(selected_families)
    for raw_entry in _load_seed_entries(seed_url_pack):
        entry = dict(raw_entry)
        entry["_snapshot_id"] = active_snapshot_id
        source_url = str(entry.get("url") or entry.get("path") or "").strip()
        if not source_url:
            continue
        if entry.get("page_type") == "handbook":
            continue
        html_text = _read_text(source_url)
        page_title = _extract_title(html_text)
        page_type = _infer_page_type(source_url, page_title, policies, explicit=entry.get("page_type"))
        source_family = _family_for_page_type(page_type)
        if source_family not in selected_families or source_family == "handbook_policy":
            continue
        text = _html_to_text(html_text)
        headings = _extract_headings(html_text)
        school_year = _school_year(entry, page_title, text)
        payloads = _extract_script_payloads(html_text)
        document_row = _build_document_row(
            entry=entry,
            source_url=source_url,
            page_type=page_type,
            page_title=page_title,
            text=text,
            school_year=school_year,
            raw_payload={"script_payload_count": len(payloads)},
        )
        documents.append(document_row)
        family_chunks = _build_chunk_rows(document_row, headings)
        chunks.extend(family_chunks)
        _record_family_metric(family_metadata, source_family, "documents", 1)
        _record_family_metric(family_metadata, source_family, "chunks", len(family_chunks))
        if page_type == "course":
            extracted_courses = _course_rows_from_json(
                payloads,
                entry=entry,
                page_title=page_title,
                source_url=source_url,
                school_year=school_year,
            )
            if not extracted_courses:
                extracted_courses = _course_rows_from_text(
                    text,
                    entry=entry,
                    page_title=page_title,
                    source_url=source_url,
                    school_year=school_year,
                )
            courses.extend(extracted_courses)
            _record_family_metric(family_metadata, source_family, "course_versions", len(extracted_courses))
        elif page_type == "faculty":
            extracted_faculty = _faculty_rows_from_payloads(
                payloads,
                entry=entry,
                page_title=page_title,
                source_url=source_url,
            )
            if not extracted_faculty:
                extracted_faculty = _faculty_rows_from_text(
                    text,
                    entry=entry,
                    page_title=page_title,
                    source_url=source_url,
                )
            faculty.extend(extracted_faculty)
            _record_family_metric(family_metadata, source_family, "faculty_profiles", len(extracted_faculty))
        elif page_type in {"admissions", "admissions_general"}:
            extracted_admissions = _admissions_rows_from_text(
                text,
                entry=entry,
                page_title=page_title,
                source_url=source_url,
            )
            admissions.extend(extracted_admissions)
            _record_family_metric(family_metadata, source_family, "admissions_facts", len(extracted_admissions))
        elif page_type == "publication":
            extracted_publications = _publication_rows_from_text(
                text,
                entry=entry,
                page_title=page_title,
                source_url=source_url,
                school_year=school_year,
            )
            publications.extend(extracted_publications)
            _record_family_metric(
                family_metadata,
                source_family,
                "publication_versions",
                len(extracted_publications),
            )
        elif page_type == "athletics":
            extracted_teams, extracted_games, extracted_records = _athletics_rows_from_payloads(
                payloads,
                entry=entry,
                page_title=page_title,
                source_url=source_url,
                school_year=school_year,
            )
            athletics_teams.extend(extracted_teams)
            athletics_games.extend(extracted_games)
            athletics_records.extend(extracted_records)
            _record_family_metric(family_metadata, source_family, "athletics_teams", len(extracted_teams))
            _record_family_metric(family_metadata, source_family, "athletics_games", len(extracted_games))
            _record_family_metric(family_metadata, source_family, "athletics_records", len(extracted_records))
    store.upsert_source_documents(documents)
    store.upsert_retrieval_chunks(chunks)
    if courses:
        store.upsert_course_versions(courses)
    if faculty:
        store.upsert_faculty_profiles(faculty)
    if admissions:
        store.upsert_admissions_facts(admissions)
    if publications:
        store.upsert_publication_versions(publications)
    if athletics_teams:
        store.upsert_athletics_teams(athletics_teams)
    if athletics_games:
        store.upsert_athletics_games(athletics_games)
    if athletics_records:
        store.upsert_athletics_records(athletics_records)
    snapshot = None
    if complete_snapshot:
        snapshot = store.complete_snapshot(active_snapshot_id, metadata={"families": family_metadata})
    return {
        "snapshot_id": active_snapshot_id,
        "documents_ingested": len(documents),
        "chunks_ingested": len(chunks),
        "course_versions_ingested": len(courses),
        "faculty_profiles_ingested": len(faculty),
        "admissions_facts_ingested": len(admissions),
        "publication_versions_ingested": len(publications),
        "athletics_teams_ingested": len(athletics_teams),
        "athletics_games_ingested": len(athletics_games),
        "athletics_records_ingested": len(athletics_records),
        "families": family_metadata,
        "completed": complete_snapshot,
        "snapshot_metadata": snapshot.extra if snapshot is not None else None,
    }


def _extract_pdf_text_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append({"page_number": index, "text": _normalize_space(page.extract_text() or "")})
    return pages


def _extract_pdf_pages_via_ocr(pdf_bytes: bytes) -> list[dict[str, Any]]:
    del pdf_bytes
    return []


def _native_text_quality(pages: list[dict[str, Any]]) -> float:
    text = " ".join(page.get("text", "") for page in pages)
    if not text:
        return 0.0
    alpha_chars = sum(1 for char in text if char.isalpha())
    return alpha_chars / max(len(text), 1)


def _split_handbook_sections(
    pages: list[dict[str, Any]],
    *,
    snapshot_id: str,
    handbook_url: str,
    handbook_version: str | None,
    extraction_quality: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page.get("page_number", 0))
        text = _normalize_space(page.get("text", ""))
        if not text:
            continue
        lines = [line.strip() for line in page.get("text", "").splitlines() if line.strip()]
        section_title = _normalize_space(lines[0] if lines else f"Page {page_number}")
        section_key = _normalize_key(f"{handbook_version or 'handbook'}-{section_title}-{page_number}")
        citation_label = f"{handbook_version or 'Handbook'}, page {page_number}"
        rows.append(
            {
                "id": _stable_id("handbook", [snapshot_id, page_number, section_title, text[:240]]),
                "snapshot_id": snapshot_id,
                "section_key": section_key,
                "handbook_version": handbook_version,
                "section_title": section_title or f"Page {page_number}",
                "section_path": section_title or f"Page {page_number}",
                "page_start": page_number,
                "page_end": page_number,
                "text": text,
                "source_url": handbook_url,
                "citation_label": citation_label,
                "content_hash": _stable_id("hash", [handbook_version, page_number, text]),
                "extraction_quality": extraction_quality,
                "extra": {"page_number": page_number},
                "current": False,
            }
        )
    return rows


def _extract_marked_text_pages(text: str) -> list[dict[str, Any]]:
    markers = list(re.finditer(r"(?m)^===\s*Page\s+(\d+)\s*===\s*$", text))
    if not markers:
        return []
    pages: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        pages.append({"page_number": int(marker.group(1)), "text": chunk})
    return pages


def ingest_webb_handbook(
    dsn: str,
    handbook_url: str,
    *,
    snapshot_id: str | None = None,
    label: str = "webb-handbook-sync",
    allow_ocr_fallback: bool = False,
    complete_snapshot: bool = True,
) -> dict[str, Any]:
    store = WebbKnowledgeStore(dsn)
    store.create_schema()
    family_metadata = _family_metadata_template(["handbook_policy"])
    active_snapshot_id = snapshot_id or store.create_snapshot(
        label=label,
        handbook_url=handbook_url,
        metadata={"created_at": datetime.now(timezone.utc).isoformat()},
    )
    handbook_path = Path(handbook_url)
    if handbook_path.exists() and handbook_path.suffix.lower() in {".txt", ".md"}:
        text = _read_text(handbook_url)
        pages = _extract_marked_text_pages(text)
        if not pages:
            pages = [
                {"page_number": index + 1, "text": chunk}
                for index, chunk in enumerate(
                    [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
                    or _chunk_text(text, heading=None, size=1200)
                )
            ]
        extraction_quality = "native_text"
    else:
        pdf_bytes = _read_bytes(handbook_url)
        pages = _extract_pdf_text_pages(pdf_bytes)
        extraction_quality = "native_text" if _native_text_quality(pages) >= 0.60 else "degraded"
        if extraction_quality == "degraded" and allow_ocr_fallback:
            ocr_pages = _extract_pdf_pages_via_ocr(pdf_bytes)
            if _native_text_quality(ocr_pages) > _native_text_quality(pages):
                pages = ocr_pages
                extraction_quality = "ocr"
    handbook_version = _school_year({"school_year": None}, handbook_url, " ".join(page["text"] for page in pages))
    sections = _split_handbook_sections(
        pages,
        snapshot_id=active_snapshot_id,
        handbook_url=handbook_url,
        handbook_version=handbook_version,
        extraction_quality=extraction_quality,
    )
    store.upsert_handbook_sections(sections)
    _record_family_metric(family_metadata, "handbook_policy", "handbook_sections", len(sections))
    snapshot = None
    if complete_snapshot:
        snapshot = store.complete_snapshot(
            active_snapshot_id,
            metadata={
                "handbook_extraction_quality": extraction_quality,
                "families": family_metadata,
            },
        )
    return {
        "snapshot_id": active_snapshot_id,
        "handbook_sections_ingested": len(sections),
        "extraction_quality": extraction_quality,
        "families": family_metadata,
        "completed": complete_snapshot,
        "snapshot_metadata": snapshot.extra if snapshot is not None else None,
    }


def webb_sync(
    dsn: str,
    *,
    seed_url_pack: str,
    source_policy_path: str | None = None,
    handbook_url: str | None = None,
    allow_ocr_fallback: bool = False,
    label: str = "webb-sync",
    families: list[str] | None = None,
) -> dict[str, Any]:
    store = WebbKnowledgeStore(dsn)
    store.create_schema()
    selected_families = _normalize_selected_families(families)
    if not handbook_url and "handbook_policy" in selected_families:
        selected_families = [family for family in selected_families if family != "handbook_policy"]
    previous_snapshot = store.latest_completed_snapshot()
    snapshot_id = store.create_snapshot(
        label=label,
        seed_url_pack=seed_url_pack,
        handbook_url=handbook_url,
        metadata={"created_at": datetime.now(timezone.utc).isoformat()},
    )
    site_result = None
    site_families = [family for family in selected_families if family != "handbook_policy"]
    if site_families:
        site_result = ingest_webb_site(
            dsn,
            seed_url_pack,
            source_policy_path=source_policy_path,
            snapshot_id=snapshot_id,
            label=label,
            families=site_families,
            complete_snapshot=False,
        )
    handbook_result = None
    if handbook_url and "handbook_policy" in selected_families:
        handbook_result = ingest_webb_handbook(
            dsn,
            handbook_url,
            snapshot_id=snapshot_id,
            allow_ocr_fallback=allow_ocr_fallback,
            label=label,
            complete_snapshot=False,
        )
    carried_forward_families = [family for family in ALL_WEBB_FAMILIES if family not in selected_families]
    carry_forward_counts = store.carry_forward_families(
        previous_snapshot.id if previous_snapshot is not None else None,
        snapshot_id,
        families=carried_forward_families,
    )
    family_metadata = _family_metadata_template(ALL_WEBB_FAMILIES)
    if site_result:
        for family, payload in (site_result.get("families") or {}).items():
            family_metadata[family].update(payload)
    if handbook_result:
        for family, payload in (handbook_result.get("families") or {}).items():
            family_metadata[family].update(payload)
    for family in carried_forward_families:
        if carry_forward_counts.get(family, 0) <= 0:
            continue
        family_metadata[family]["carried_forward"] = True
        family_metadata[family]["refreshed"] = False
        family_metadata[family]["rows"] = {
            **dict(family_metadata[family].get("rows") or {}),
            "carried_forward_rows": int(carry_forward_counts.get(family, 0)),
        }
    snapshot = store.complete_snapshot(
        snapshot_id,
        metadata={
            "handbook_extraction_quality": (handbook_result or {}).get("extraction_quality"),
            "families": family_metadata,
            "selected_families": selected_families,
            "carried_forward_families": carried_forward_families,
            "freshness_policy": DEFAULT_FAMILY_FRESHNESS,
            "source_policy_path": source_policy_path,
        },
    )
    return {
        "snapshot_id": snapshot_id,
        "site": site_result,
        "handbook": handbook_result,
        "selected_families": selected_families,
        "carried_forward_families": carried_forward_families,
        "carry_forward_counts": carry_forward_counts,
        "families": family_metadata,
        "snapshot_metadata": snapshot.extra,
        "quality": store.snapshot_quality(snapshot_id),
    }


def diff_webb_snapshot(dsn: str, snapshot_from: str, snapshot_to: str) -> dict[str, Any]:
    store = WebbKnowledgeStore(dsn)
    diff = store.diff_snapshots(snapshot_from, snapshot_to)
    before = {row.source_url: row for row in store.list_source_documents(snapshot_from)}
    after = {row.source_url: row for row in store.list_source_documents(snapshot_to)}
    changed_families: dict[str, dict[str, int]] = {}
    for url in diff["added_urls"]:
        family = after[url].source_family
        bucket = changed_families.setdefault(family, {"added": 0, "removed": 0, "changed": 0, "unchanged": 0})
        bucket["added"] += 1
    for url in diff["removed_urls"]:
        family = before[url].source_family
        bucket = changed_families.setdefault(family, {"added": 0, "removed": 0, "changed": 0, "unchanged": 0})
        bucket["removed"] += 1
    for url in diff["changed_urls"]:
        family = after.get(url, before.get(url)).source_family
        bucket = changed_families.setdefault(family, {"added": 0, "removed": 0, "changed": 0, "unchanged": 0})
        bucket["changed"] += 1
    for url in diff["unchanged_urls"]:
        family = after.get(url, before.get(url)).source_family
        bucket = changed_families.setdefault(family, {"added": 0, "removed": 0, "changed": 0, "unchanged": 0})
        bucket["unchanged"] += 1
    diff["changed_families"] = changed_families
    return diff
