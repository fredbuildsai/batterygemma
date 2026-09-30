"""OpenAlex works search: open-access articles, preprints and dissertations with an allowed license."""

import logging
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlencode

from batterygemma.sources.base import DiscoveredRecord, PoliteClient, normalize_doi

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org/works"
MAX_PER_PAGE = 200
# The curated field set `_to_record` actually reads - the default, to keep response payloads small (see
# OpenAlexSource's `fields` param for how to get a custom subset or the full, unrestricted response instead).
DEFAULT_FIELDS = [
    "id", "doi", "title", "publication_year", "type", "language", "authorships", "abstract_inverted_index",
    "best_oa_location", "primary_location", "open_access",
]
SELECT = ",".join(DEFAULT_FIELDS)  # kept for backward compatibility with anything importing the old constant


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
        fields: list[str] | None = DEFAULT_FIELDS,
    ) -> None:
        """`fields`: which OpenAlex work fields to request via `select=`.

        - Omitted (default): `DEFAULT_FIELDS`, the curated subset `_to_record` reads - keeps responses small.
        - A custom list: exactly those fields (must include whatever `_to_record` needs if you still want a
          populated `DiscoveredRecord` back from `discover()`).
        - `None` (passed explicitly): no `select` param at all - the full, unrestricted OpenAlex record.
          `_to_record` still works (it only reads the fields it needs and ignores the rest), and the complete
          raw work dict is kept on `DiscoveredRecord.raw` instead of the usual 3-key summary, so nothing
          extra OpenAlex returned is thrown away.
        """
        self.client = client
        self.licenses = licenses
        self.contact_email = contact_email
        self.api_key = api_key if api_key is not None else os.environ.get("OPENALEX_API_KEY")
        self.per_page = min(per_page, MAX_PER_PAGE)
        self.fields = fields

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
                params = self._params(term, cursor, min(self.per_page, per_term))
                # Safe to log the full URL now: the API key travels as an Authorization header (see
                # _auth_headers), never in the query string, so nothing sensitive ends up in this line -
                # unlike the old api_key-as-query-param approach, where logging this would have leaked it.
                logger.info("GET %s?%s", BASE_URL, urlencode(params),
                           extra={"context": {"source": self.name, "term": term}})
                page = self.client.get(BASE_URL, params=params, headers=self._auth_headers()).json()
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
        }
        if self.fields:  # None or [] -> omit `select` entirely, i.e. request the full record
            params["select"] = ",".join(self.fields)
        if self.contact_email:
            params["mailto"] = self.contact_email
        return params

    def _auth_headers(self) -> dict[str, str] | None:
        """The API key as a Bearer header, not a query parameter - a query-string secret is far more likely
        to end up in a proxy's or CDN's access log (which commonly logs full URLs, rarely header values) or
        used as a cache key somewhere, than a header is. `mailto` stays a query param deliberately: it is
        OpenAlex's documented, non-secret "polite pool" identification mechanism, not a credential."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None

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
            # When `fields` requested the full record (self.fields is falsy), keep all of it here rather
            # than just the usual 3-key summary - nothing OpenAlex returned is thrown away in that mode.
            raw=work if not self.fields else {
                "type": work.get("type"), "language": work.get("language"),
                "oa_status": (work.get("open_access") or {}).get("oa_status"),
            },
        )
