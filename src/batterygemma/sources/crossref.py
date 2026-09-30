"""ChemRxiv discovery through Crossref metadata (DOI prefix 10.26434).

ChemRxiv's own API and PDF links sit behind a Cloudflare bot challenge, so metadata (title, abstract,
license, links) comes from Crossref. Versioned DOIs ("…/v2", "….v3") often lack the license field, so
versions are merged onto their base DOI and the license is taken from whichever version carries it.
"""

import logging
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlencode

from batterygemma.sources.base import DiscoveredRecord, PoliteClient, normalize_doi

logger = logging.getLogger(__name__)

BASE_URL = "https://api.crossref.org/works"
CHEMRXIV_PREFIX = "10.26434"
DEFAULT_FIELDS = ["DOI", "title", "abstract", "author", "license", "link", "posted", "URL"]
SELECT = ",".join(DEFAULT_FIELDS)  # kept for backward compatibility with anything importing the old constant
_VERSION = re.compile(r"(/v\d+|\.v\d+)$")


def base_doi(doi: str) -> str:
    return _VERSION.sub("", doi)


def strip_markup(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip() or None


class CrossrefChemRxivSource:
    name = "chemrxiv"

    def __init__(
        self, client: PoliteClient, *, rows: int = 100, contact_email: str = "",
        fields: list[str] | None = DEFAULT_FIELDS,
    ) -> None:
        """`fields`: see `OpenAlexSource.__init__`'s docstring - same three modes (default curated subset,
        a custom list, or `None` for the full unrestricted Crossref record)."""
        self.client = client
        self.rows = rows
        self.contact_email = contact_email
        self.fields = fields

    def discover(self, terms: list[str], limit: int) -> Iterator[DiscoveredRecord]:
        merged: dict[str, DiscoveredRecord] = {}
        for term in terms:
            cursor: str | None = "*"
            while cursor and len(merged) < limit:
                params = self._params(term, cursor)
                logger.info("GET %s?%s", BASE_URL, urlencode(params), extra={"context": {"source": self.name, "term": term}})
                message = self.client.get(BASE_URL, params=params).json()["message"]
                items = message.get("items", [])
                for item in items:
                    self._merge(merged, item)
                cursor = message.get("next-cursor") if items else None
        yield from list(merged.values())[:limit]

    def _params(self, term: str, cursor: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "filter": f"prefix:{CHEMRXIV_PREFIX},type:posted-content",
            "query.bibliographic": term,
            "rows": self.rows,
            "cursor": cursor,
        }
        if self.fields:  # None or [] -> omit `select` entirely, i.e. request the full record
            params["select"] = ",".join(self.fields)
        if self.contact_email:
            params["mailto"] = self.contact_email
        return params

    def _merge(self, merged: dict[str, DiscoveredRecord], item: dict[str, Any]) -> None:
        doi = normalize_doi(item.get("DOI"))
        titles = item.get("title") or []
        if not doi or not titles:
            return
        key = base_doi(doi)
        license_url = next((lic.get("URL") for lic in item.get("license", []) if lic.get("URL")), None)
        pdf_url = next((link["URL"] for link in item.get("link", []) if "/pdf/" in link.get("URL", "")), None)
        posted = (item.get("posted") or {}).get("date-parts") or [[None]]

        record = merged.get(key)
        if record is None:
            record = DiscoveredRecord(
                source=self.name,
                external_id=key.removeprefix(f"{CHEMRXIV_PREFIX}/"),
                title=titles[0],
                doi=key,
                authors=[" ".join(p for p in (a.get("given"), a.get("family")) if p) for a in item.get("author", [])],
                year=posted[0][0],
                abstract=strip_markup(item.get("abstract")),
                url=f"https://doi.org/{key}",
                raw={"versions": []},
            )
            merged[key] = record
        record.raw["versions"].append(doi)
        if not self.fields:
            # Full-record mode: keep every version's complete raw item too, alongside the existing
            # "versions" DOI list - additive, doesn't disturb the version-merge tracking above.
            record.raw.setdefault("full", {})[doi] = item
        if license_url and not record.license_raw:
            record.license_raw = license_url
            record.license_evidence = f"crossref:{doi}:license.URL"
        if pdf_url and (doi != key or not record.pdf_url):  # prefer the latest versioned PDF link
            record.pdf_url = pdf_url
        if not record.abstract:
            record.abstract = strip_markup(item.get("abstract"))
