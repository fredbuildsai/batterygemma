"""OpenAlex works search: open-access articles, preprints and dissertations with an allowed license."""

import os
from collections.abc import Iterator
from typing import Any

from batterygemma.sources.base import DiscoveredRecord, PoliteClient, normalize_doi

BASE_URL = "https://api.openalex.org/works"
MAX_PER_PAGE = 200
SELECT = ",".join(
    ["id", "doi", "title", "publication_year", "type", "language", "authorships", "abstract_inverted_index",
     "best_oa_location", "primary_location", "open_access"]
)


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    positions = sorted((pos, word) for word, places in inverted_index.items() for pos in places)
    return " ".join(word for _, word in positions)


class OpenAlexSource:
    name = "openalex"

    def __init__(
        self,
        client: PoliteClient,
        *,
        licenses: tuple[str, ...] = ("cc-by", "cc0", "public-domain"),
        contact_email: str = "",
        api_key: str | None = None,
        per_page: int = MAX_PER_PAGE,
    ) -> None:
        self.client = client
        self.licenses = licenses
        self.contact_email = contact_email
        self.api_key = api_key if api_key is not None else os.environ.get("OPENALEX_API_KEY")
        self.per_page = min(per_page, MAX_PER_PAGE)

    def discover(self, terms: list[str], limit: int) -> Iterator[DiscoveredRecord]:
        """Yield up to `limit` unique works, split evenly across `terms`.

        Without the per-term cap, a broad first query ("lithium-ion battery", ~100k hits) would fill the whole
        limit and the topic-specific queries (cathodes, electrolytes, SEI, ...) would contribute nothing.
        """
        seen: set[str] = set()
        per_term = max(1, -(-limit // max(1, len(terms))))  # ceiling division
        for term in terms:
            taken = 0
            cursor: str | None = "*"
            while cursor and taken < per_term and len(seen) < limit:
                page = self.client.get(BASE_URL, params=self._params(term, cursor, min(self.per_page, per_term))).json()
                results = page.get("results", [])
                for work in results:
                    record = self._to_record(work)
                    if record is None or record.external_id in seen:
                        continue
                    seen.add(record.external_id)
                    taken += 1
                    yield record
                    if taken >= per_term or len(seen) >= limit:
                        break
                cursor = page.get("meta", {}).get("next_cursor") if results else None
            if len(seen) >= limit:
                return

    def _params(self, term: str, cursor: str, per_page: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "search": term,
            "filter": f"is_oa:true,best_oa_location.license:{'|'.join(self.licenses)}",
            "per-page": per_page,
            "cursor": cursor,
            "select": SELECT,
        }
        if self.contact_email:
            params["mailto"] = self.contact_email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _to_record(self, work: dict[str, Any]) -> DiscoveredRecord | None:
        title = work.get("title")
        if not title:
            return None
        best = work.get("best_oa_location") or {}
        primary = work.get("primary_location") or {}
        venue = ((primary.get("source") or {}).get("display_name")) or None
        return DiscoveredRecord(
            source=self.name,
            external_id=work["id"].rsplit("/", 1)[-1],
            title=title,
            doi=normalize_doi(work.get("doi")),
            authors=[a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")],
            year=work.get("publication_year"),
            venue=venue,
            abstract=rebuild_abstract(work.get("abstract_inverted_index")),
            url=best.get("landing_page_url") or primary.get("landing_page_url"),
            pdf_url=best.get("pdf_url"),
            license_raw=best.get("license"),
            license_evidence="openalex:best_oa_location.license",
            raw={"type": work.get("type"), "language": work.get("language"), "oa_status": (work.get("open_access") or {}).get("oa_status")},
        )
