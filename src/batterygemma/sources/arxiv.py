"""arXiv discovery via OAI-PMH (arXivRaw), the bulk-harvesting interface arXiv recommends.

arXivRaw carries the per-paper <license> URL, which the search API does not. Records are harvested
per set and date window, then filtered locally by category and scope keywords.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from email.utils import parsedate_to_datetime

from batterygemma.sources.base import DiscoveredRecord, PoliteClient

BASE_URL = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return re.sub(r"\s+", " ", element.text).strip() or None


class ArxivOaiSource:
    name = "arxiv"

    def __init__(
        self,
        client: PoliteClient,
        *,
        sets: list[str],
        categories: list[str],
        from_date: str | None = None,
        until_date: str | None = None,
    ) -> None:
        self.client = client
        self.sets = sets
        self.categories = set(categories)
        self.from_date = from_date
        self.until_date = until_date

    def discover(self, terms: list[str], limit: int) -> Iterator[DiscoveredRecord]:
        patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in terms]
        found = 0
        for set_spec in self.sets:
            token: str | None = None
            while True:
                params = {"verb": "ListRecords", "resumptionToken": token} if token else self._initial_params(set_spec)
                root = ET.fromstring(self.client.get(BASE_URL, params=params).content)
                error = root.find("oai:error", NS)
                if error is not None:
                    if error.get("code") == "noRecordsMatch":
                        break
                    raise RuntimeError(f"arXiv OAI-PMH error {error.get('code')}: {error.text}")
                for record in root.iterfind("oai:ListRecords/oai:record", NS):
                    parsed = self._to_record(record)
                    if parsed and self._in_scope(parsed, patterns):
                        yield parsed
                        found += 1
                        if found >= limit:
                            return
                token = _text(root.find("oai:ListRecords/oai:resumptionToken", NS))
                if not token:
                    break

    def _initial_params(self, set_spec: str) -> dict[str, str]:
        params = {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": set_spec}
        if self.from_date:
            params["from"] = self.from_date
        if self.until_date:
            params["until"] = self.until_date
        return params

    def _in_scope(self, record: DiscoveredRecord, patterns: list[re.Pattern[str]]) -> bool:
        if self.categories and not self.categories & set(record.raw.get("categories", [])):
            return False
        haystack = f"{record.title} {record.abstract or ''}"
        return any(p.search(haystack) for p in patterns)

    def _to_record(self, record: ET.Element) -> DiscoveredRecord | None:
        if record.find("oai:header", NS) is not None and record.find("oai:header", NS).get("status") == "deleted":
            return None
        raw = record.find("oai:metadata/raw:arXivRaw", NS)
        if raw is None:
            return None
        arxiv_id = _text(raw.find("raw:id", NS))
        title = _text(raw.find("raw:title", NS))
        if not arxiv_id or not title:
            return None
        year = None
        first_version_date = _text(raw.find("raw:version/raw:date", NS))
        if first_version_date:
            try:
                year = parsedate_to_datetime(first_version_date).year
            except (TypeError, ValueError):
                year = None
        authors = _text(raw.find("raw:authors", NS)) or ""
        license_url = _text(raw.find("raw:license", NS))
        return DiscoveredRecord(
            source=self.name,
            external_id=arxiv_id,
            title=title,
            doi=(_text(raw.find("raw:doi", NS)) or "").lower() or None,
            authors=[a.strip() for a in re.split(r",|\band\b", authors) if a.strip()],
            year=year,
            abstract=_text(raw.find("raw:abstract", NS)),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            license_raw=license_url,
            license_evidence="arxiv:oai-pmh:arXivRaw.license" if license_url else None,
            raw={"categories": (_text(raw.find("raw:categories", NS)) or "").split()},
        )
