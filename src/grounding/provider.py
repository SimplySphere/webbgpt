from __future__ import annotations

import re
from typing import Any

from grounding.store import CatalogStore, WebbKnowledgeStore
from grounding.types import Citation, GroundingHit, GroundingResult, RouteDecision


COURSE_CODE_RE = re.compile(r"\b([a-z]{2,8})\s?-?(\d{2,4}[a-z]?)\b", re.IGNORECASE)
SCHOOL_YEAR_RE = re.compile(r"(20\d{2})\s*[-–]\s*(\d{2}|\d{4})")


class GroundingProvider:
    def __init__(self, store: CatalogStore):
        self.store = store

    def query(self, text: str, institution_id: str | None = None, limit: int = 5) -> GroundingResult:
        terms: list[str] = []
        for match in COURSE_CODE_RE.finditer(text):
            subject, number = match.groups()
            normalized = f"{subject.upper()} {number.upper()}"
            compact = f"{subject.upper()}{number.upper()}"
            if normalized not in terms:
                terms.append(normalized)
            if compact not in terms:
                terms.append(compact)
        if text not in terms:
            terms.append(text)

        courses = []
        seen_course_ids: set[str] = set()
        for term in terms:
            for course in self.store.search_courses(term, institution_id=institution_id, limit=limit):
                if course.id in seen_course_ids:
                    continue
                seen_course_ids.add(course.id)
                courses.append(course)
                if len(courses) >= limit:
                    break
            if len(courses) >= limit:
                break
        hits: list[GroundingHit] = []
        for course in courses:
            content = course.description or ""
            citations = [
                Citation(
                    source_type="course",
                    source_id=course.id,
                    label=course.code,
                    snippet=content[:280],
                    metadata={"title": course.title, "credits": course.credits},
                )
            ]
            hits.append(
                GroundingHit(
                    title=f"{course.code}: {course.title}",
                    content=content,
                    citations=citations,
                    metadata={"course_id": course.id, "program_id": course.program_id},
                )
            )
        return GroundingResult(query=text, hits=hits, route="course_catalog", metadata={"mode": "legacy_catalog"})


class WebbGroundingProvider:
    ROUTE_ALIASES = {
        "handbook": "handbook_policy",
        "admissions": "admissions_general",
    }
    GROUNDED_ROUTES = {
        "course_catalog",
        "faculty",
        "handbook_policy",
        "student_life",
        "admissions_general",
        "museum_programs",
        "athletics",
        "planner_advice",
    }
    MANDATORY_GROUNDED_ROUTES = {
        "course_catalog",
        "faculty",
        "handbook_policy",
        "athletics",
    }

    def __init__(
        self,
        store: WebbKnowledgeStore,
        *,
        snapshot_id: str | None = "latest",
        route_fanout_limit: int = 2,
        planner_beta_enabled: bool = False,
    ):
        self.store = store
        self.snapshot_id = snapshot_id
        self.route_fanout_limit = max(int(route_fanout_limit), 1)
        self.planner_beta_enabled = planner_beta_enabled

    @classmethod
    def normalize_route(cls, route: str | None) -> str:
        if route is None:
            return "chat"
        normalized = str(route).strip()
        return cls.ROUTE_ALIASES.get(normalized, normalized)

    def _extract_school_years(self, text: str) -> list[str]:
        years: list[str] = []
        for left, right in SCHOOL_YEAR_RE.findall(text):
            normalized_right = right if len(right) == 4 else f"{left[:2]}{right}"
            years.append(f"{left}-{normalized_right[-2:]}")
        return years

    def route_messages(self, messages: list[Any]) -> RouteDecision:
        latest_user = next(
            (str(message.content if hasattr(message, "content") else message.get("content", "")) for message in reversed(messages) if (getattr(message, "role", None) or message.get("role")) == "user"),
            "",
        )
        lowered = latest_user.lower()
        school_years = self._extract_school_years(latest_user)
        scores = {
            "course_catalog": 0,
            "faculty": 0,
            "handbook_policy": 0,
            "student_life": 0,
            "admissions_general": 0,
            "museum_programs": 0,
            "athletics": 0,
            "planner_advice": 0,
        }
        if any(keyword in lowered for keyword in ["handbook", "policy", "attendance", "academic honesty", "phone", "rule", "study hall", "residential expectation"]):
            scores["handbook_policy"] += 5
        if any(keyword in lowered for keyword in ["faculty", "teacher", "teachers", "dean", "who teaches", "who is", "directory", "staff"]):
            scores["faculty"] += 4
        if any(keyword in lowered for keyword in ["apply", "application", "admission", "admissions", "tuition", "financial aid", "deadline", "mission", "values", "community", "college guidance", "prospective", "interview"]):
            scores["admissions_general"] += 4
        if any(keyword in lowered for keyword in ["dorm", "dorm life", "dining", "food", "chapel", "clubs", "affinity", "student leadership", "wellness", "weekend", "student life", "what is webb like", "culture"]):
            scores["student_life"] += 4
        if any(keyword in lowered for keyword in ["museum", "alf", "unique learning", "unique program", "paleontology museum", "research opportunity"]):
            scores["museum_programs"] += 4
        if any(keyword in lowered for keyword in ["athletics", "team", "schedule", "record", "won", "lost", "game", "season", "coach", "tennis", "soccer", "basketball", "softball", "volleyball"]):
            scores["athletics"] += 4
        if any(keyword in lowered for keyword in ["overwhelmed", "plan my week", "weekly plan", "time management", "procrastinate", "study plan", "balance sports", "routine", "catch up this week"]):
            scores["planner_advice"] += 5
        if COURSE_CODE_RE.search(latest_user):
            scores["course_catalog"] += 5
        if any(
            keyword in lowered
            for keyword in [
                "course",
                "catalog",
                "prereq",
                "prerequisite",
                "upperclassmen",
                "junior",
                "senior",
                "offering",
                "offerings",
                "classes",
                "math",
                "science",
                "department",
                "change from",
                "difference between",
                "compare classes",
            ]
        ):
            scores["course_catalog"] += 3
        ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        top_route, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0
        if top_score <= 0:
            return RouteDecision(
                route="chat",
                query=latest_user,
                grounded=False,
                school_years=school_years,
                metadata={"route_scores": scores, "candidate_routes": [], "fanout_routes": [], "low_confidence": False},
            )
        low_confidence = top_score < 4 or (second_score > 0 and (top_score - second_score) <= 1)
        candidate_routes = [route for route, score in ordered if score > 0]
        fanout_routes = candidate_routes[: self.route_fanout_limit] if low_confidence else [top_route]
        grounded = top_route in self.GROUNDED_ROUTES and not (
            top_route == "planner_advice" and not self.planner_beta_enabled
        )
        return RouteDecision(
            route=top_route,
            query=latest_user,
            grounded=grounded,
            school_years=school_years,
            metadata={
                "route_scores": scores,
                "candidate_routes": candidate_routes,
                "fanout_routes": fanout_routes,
                "low_confidence": low_confidence,
                "planner_beta_enabled": self.planner_beta_enabled,
                "planner_beta_disabled": top_route == "planner_advice" and not self.planner_beta_enabled,
            },
        )

    @staticmethod
    def _citation(source_type: str, source_id: str, label: str, snippet: str, metadata: dict[str, Any]) -> Citation:
        return Citation(
            source_type=source_type,
            source_id=source_id,
            label=label,
            snippet=snippet[:280],
            metadata=metadata,
        )

    def _course_hits(self, query: str, school_years: list[str], limit: int) -> list[GroundingHit]:
        target_year = school_years[-1] if len(school_years) == 1 else None
        rows = self.store.search_course_versions(query, snapshot_id=self.snapshot_id, school_year=target_year, limit=limit)
        hits: list[GroundingHit] = []
        for row in rows:
            content_parts = [row.description or ""]
            if row.prerequisites:
                content_parts.append(f"Prerequisites: {row.prerequisites}")
            if row.teacher_names and row.teacher_names.get("names"):
                content_parts.append(f"Teachers: {', '.join(row.teacher_names['names'])}")
            content = "\n".join(part for part in content_parts if part).strip()
            hits.append(
                GroundingHit(
                    title=f"{row.course_code or row.title} ({row.school_year or 'current'})",
                    content=content,
                    citations=[
                        self._citation(
                            "course",
                            row.id,
                            row.citation_label,
                            content,
                            {"school_year": row.school_year, "department": row.department, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "course_catalog", "school_year": row.school_year, "source_url": row.source_url},
                )
            )
        return hits

    def _publication_hits(self, query: str, limit: int) -> list[GroundingHit]:
        rows = self.store.search_publications(query, snapshot_id=self.snapshot_id, limit=limit)
        hits: list[GroundingHit] = []
        for row in rows:
            hits.append(
                GroundingHit(
                    title=f"{row.title} ({row.school_year or 'current'})",
                    content=row.text,
                    citations=[
                        self._citation(
                            "publication",
                            row.id,
                            row.citation_label,
                            row.text,
                            {"school_year": row.school_year, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "course_catalog", "mode": "publication_context", "school_year": row.school_year},
                )
            )
        return hits

    def _course_diff_hits(self, query: str, school_years: list[str]) -> list[GroundingHit]:
        if len(school_years) < 2:
            return []
        diff = self.store.compare_course_versions(
            query,
            school_year_a=school_years[0],
            school_year_b=school_years[1],
            snapshot_id=self.snapshot_id,
        )
        if not any(diff.values()):
            return []
        sections: list[str] = []
        citations: list[Citation] = []
        for label, rows in (("Added", diff["added"]), ("Removed", diff["removed"]), ("Changed", diff["changed"])):
            if not rows:
                continue
            sections.append(f"{label}: " + "; ".join(f"{row.course_code or row.title}" for row in rows))
            for row in rows[:3]:
                citations.append(
                    self._citation(
                        "publication" if not row.course_code else "course",
                        row.id,
                        row.citation_label,
                        row.description or row.title,
                        {"school_year": row.school_year, "source_url": row.source_url},
                    )
                )
        return [
            GroundingHit(
                title=f"Course changes between {school_years[0]} and {school_years[1]}",
                content="\n".join(sections),
                citations=citations[:5],
                metadata={"route": "course_catalog", "mode": "year_diff", "school_years": school_years[:2]},
            )
        ]

    def _faculty_hits(self, query: str, limit: int) -> list[GroundingHit]:
        rows = self.store.search_faculty_profiles(query, snapshot_id=self.snapshot_id, limit=limit)
        hits: list[GroundingHit] = []
        for row in rows:
            content = "\n".join(part for part in [row.role_title or "", row.department or "", row.bio or ""] if part).strip()
            hits.append(
                GroundingHit(
                    title=row.name,
                    content=content,
                    citations=[
                        self._citation(
                            "faculty",
                            row.id,
                            row.citation_label,
                            content,
                            {"department": row.department, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "faculty", "source_url": row.source_url},
                )
            )
        return hits

    def _handbook_hits(self, query: str, limit: int) -> list[GroundingHit]:
        rows = self.store.search_handbook_sections(query, snapshot_id=self.snapshot_id, limit=limit)
        hits: list[GroundingHit] = []
        for row in rows:
            hits.append(
                GroundingHit(
                    title=row.section_title,
                    content=row.text,
                    citations=[
                        self._citation(
                            "handbook_policy",
                            row.id,
                            row.citation_label,
                            row.text,
                            {"page_start": row.page_start, "section_path": row.section_path, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "handbook_policy", "extraction_quality": row.extraction_quality},
                )
            )
        return hits

    def _chunk_hits(self, query: str, *, route: str, families: list[str], limit: int) -> list[GroundingHit]:
        rows = self.store.search_retrieval_chunks(
            query,
            snapshot_id=self.snapshot_id,
            source_families=families,
            limit=limit,
        )
        return self._chunk_rows_to_hits(rows, route=route)

    def _chunk_rows_to_hits(self, rows, *, route: str) -> list[GroundingHit]:
        hits: list[GroundingHit] = []
        for row in rows:
            source_url = None
            page_title = row.citation_label
            if isinstance(row.extra, dict):
                page_title = str(row.extra.get("page_title") or row.citation_label)
            hits.append(
                GroundingHit(
                    title=page_title,
                    content=row.text,
                    citations=[
                        self._citation(
                            row.source_family,
                            row.id,
                            row.citation_label,
                            row.text,
                            {
                                "source_family": row.source_family,
                                "source_type": row.source_type,
                                "source_url": source_url,
                                "heading": row.heading,
                            },
                        )
                    ],
                    metadata={"route": route, "source_family": row.source_family, "source_type": row.source_type},
                )
            )
        return hits

    def _admissions_hits(self, query: str, limit: int) -> list[GroundingHit]:
        rows = self.store.search_admissions_facts(query, snapshot_id=self.snapshot_id, limit=limit)
        hits: list[GroundingHit] = []
        for row in rows:
            hits.append(
                GroundingHit(
                    title=row.title,
                    content=row.text,
                    citations=[
                        self._citation(
                            "admissions",
                            row.id,
                            row.citation_label,
                            row.text,
                            {"source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "admissions_general", "source_url": row.source_url},
                )
            )
        hits.extend(
            self._chunk_hits(
                query,
                route="admissions_general",
                families=["admissions_general", "mission_values", "college_guidance"],
                limit=limit,
            )
        )
        return self._dedupe_hits(hits, limit=limit)

    def _student_life_hits(self, query: str, limit: int) -> list[GroundingHit]:
        rows = self.store.search_retrieval_chunks(
            query,
            snapshot_id=self.snapshot_id,
            source_families=["student_life"],
            limit=max(limit * 4, 12),
        )
        lowered = query.lower()
        title_preferences = {
            "dorm": ["dorm"],
            "dining": ["dining"],
            "food": ["dining"],
            "chapel": ["chapel"],
            "club": ["clubs", "affinity"],
            "affinity": ["affinity", "clubs"],
            "leadership": ["leadership"],
            "wellness": ["wellness", "health"],
            "health": ["health", "wellness"],
            "weekend": ["after school", "weekend"],
            "after school": ["after school", "weekend"],
            "travel": ["travel"],
            "service": ["community impact"],
            "community": ["community impact"],
        }
        preferred_tokens: list[str] = []
        for trigger, tokens in title_preferences.items():
            if trigger in lowered:
                preferred_tokens.extend(tokens)
        if preferred_tokens:
            preferred = []
            fallback = []
            for row in rows:
                haystack_parts = [row.heading or "", row.citation_label or ""]
                if isinstance(row.extra, dict):
                    haystack_parts.append(str(row.extra.get("page_title") or ""))
                haystack = " ".join(haystack_parts).lower()
                if any(token in haystack for token in preferred_tokens):
                    preferred.append(row)
                else:
                    fallback.append(row)
            rows = preferred + fallback
        return self._chunk_rows_to_hits(rows[:limit], route="student_life")

    def _museum_hits(self, query: str, limit: int) -> list[GroundingHit]:
        return self._chunk_hits(query, route="museum_programs", families=["museum_programs"], limit=limit)

    def _athletics_hits(self, query: str, school_years: list[str], limit: int) -> list[GroundingHit]:
        season = school_years[-1] if school_years else None
        hits: list[GroundingHit] = []
        rows_games = self.store.search_athletics_games(query, snapshot_id=self.snapshot_id, season=season, limit=limit)
        for row in rows_games:
            content = "\n".join(
                part
                for part in [
                    f"Opponent: {row.opponent}",
                    f"Date: {row.game_date}" if row.game_date else "",
                    f"Result: {row.result}" if row.result else "",
                    f"Venue: {row.venue}" if row.venue else "",
                ]
                if part
            )
            hits.append(
                GroundingHit(
                    title=f"{row.team_name} vs {row.opponent}",
                    content=content,
                    citations=[
                        self._citation(
                            "athletics_game",
                            row.id,
                            row.citation_label,
                            content,
                            {"season": row.season, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "athletics", "season": row.season, "source_url": row.source_url},
                )
            )
        rows_records = self.store.search_athletics_records(query, snapshot_id=self.snapshot_id, season=season, limit=limit)
        for row in rows_records:
            content = f"{row.record_type}: {row.value}"
            hits.append(
                GroundingHit(
                    title=f"{row.team_name} record",
                    content=content,
                    citations=[
                        self._citation(
                            "athletics_record",
                            row.id,
                            row.citation_label,
                            content,
                            {"season": row.season, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "athletics", "season": row.season, "source_url": row.source_url},
                )
            )
        rows_teams = self.store.search_athletics_teams(query, snapshot_id=self.snapshot_id, season=season, limit=limit)
        for row in rows_teams:
            content = "\n".join(
                part
                for part in [
                    f"Coach: {row.coach_name}" if row.coach_name else "",
                    row.summary or "",
                ]
                if part
            )
            hits.append(
                GroundingHit(
                    title=row.team_name,
                    content=content,
                    citations=[
                        self._citation(
                            "athletics_team",
                            row.id,
                            row.citation_label,
                            content,
                            {"season": row.season, "source_url": row.source_url},
                        )
                    ],
                    metadata={"route": "athletics", "season": row.season, "source_url": row.source_url},
                )
            )
        if not hits:
            hits.extend(self._chunk_hits(query, route="athletics", families=["athletics"], limit=limit))
        return self._dedupe_hits(hits, limit=limit)

    def _planner_hits(self, query: str, school_years: list[str], limit: int) -> list[GroundingHit]:
        hits: list[GroundingHit] = []
        hits.extend(self._student_life_hits(query, limit=max(1, limit // 2)))
        hits.extend(self._admissions_hits(query, limit=max(1, limit // 2)))
        if any(token in query.lower() for token in ["sport", "team", "practice", "game", "athletic"]):
            hits.extend(self._athletics_hits(query, school_years, limit=max(1, limit // 2)))
        if COURSE_CODE_RE.search(query) or any(token in query.lower() for token in ["class", "course"]):
            hits.extend(self._course_hits(query, school_years, limit=max(1, limit // 2)))
        return self._dedupe_hits(hits, limit=limit)

    @staticmethod
    def _dedupe_hits(hits: list[GroundingHit], *, limit: int) -> list[GroundingHit]:
        deduped: list[GroundingHit] = []
        seen: set[str] = set()
        for hit in hits:
            citation = hit.citations[0] if hit.citations else None
            key = f"{hit.title}|{citation.source_id if citation else hit.content[:48]}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= limit:
                break
        return deduped

    def query(
        self,
        text: str,
        *,
        route: str | None = None,
        limit: int = 5,
    ) -> GroundingResult:
        school_years = self._extract_school_years(text)
        effective_route = self.normalize_route(route or self.route_messages([{"role": "user", "content": text}]).route)
        if effective_route == "course_catalog":
            wants_diff = len(school_years) >= 2 and any(
                token in text.lower() for token in ["change", "changed", "difference", "different", "compare"]
            )
            hits = self._course_diff_hits(text, school_years) if wants_diff else self._course_hits(text, school_years, limit)
            if not hits and not wants_diff:
                hits = (
                    self._course_hits(text, school_years, limit)
                    or self._publication_hits(text, limit)
                    or self._chunk_hits(text, route="course_catalog", families=["course_catalog"], limit=limit)
                )
        elif effective_route == "faculty":
            hits = self._faculty_hits(text, limit)
        elif effective_route == "handbook_policy":
            hits = self._handbook_hits(text, limit)
        elif effective_route == "admissions_general":
            hits = self._admissions_hits(text, limit)
        elif effective_route == "student_life":
            hits = self._student_life_hits(text, limit)
        elif effective_route == "museum_programs":
            hits = self._museum_hits(text, limit)
        elif effective_route == "athletics":
            hits = self._athletics_hits(text, school_years, limit)
        elif effective_route == "planner_advice":
            hits = self._planner_hits(text, school_years, limit) if self.planner_beta_enabled else []
        else:
            hits = []
        snapshot = self.store.get_snapshot(self.snapshot_id)
        return GroundingResult(
            query=text,
            hits=hits,
            route=effective_route,
            snapshot_id=None if snapshot is None else snapshot.id,
            metadata={
                "school_years": school_years,
                "grounded": effective_route != "chat",
                "planner_beta_enabled": self.planner_beta_enabled,
            },
        )

    @staticmethod
    def render_context(result: GroundingResult) -> str:
        if not result.hits:
            return ""
        blocks = []
        for hit in result.hits:
            labels = ", ".join(citation.label for citation in hit.citations[:3])
            blocks.append(f"{hit.title}\n{hit.content}\n[source: {labels}]".strip())
        return "\n\n".join(blocks)

    @staticmethod
    def no_hit_message(route: str, query: str) -> str:
        route_labels = {
            "course_catalog": "course catalog",
            "handbook_policy": "handbook",
            "faculty": "faculty directory",
            "admissions_general": "admissions, mission, and college guidance pages",
            "student_life": "student-life pages",
            "museum_programs": "museum and unique-program pages",
            "athletics": "athletics pages",
            "planner_advice": "grounded Webb support sources",
        }
        domain = route_labels.get(WebbGroundingProvider.normalize_route(route), "grounded Webb sources")
        return f"I could not find support for that in the current {domain} snapshot, so I cannot verify: {query}"
