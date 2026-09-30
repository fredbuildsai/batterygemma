"""Shared pieces for source adapters: the discovered-record shape and a polite HTTP client."""

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from batterygemma import __version__

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Bot-management signatures seen in practice while fetching OA full text (recorded, never solved/bypassed):
#   - Cloudflare: 403 with a `cf-mitigated: challenge` response header (checked separately, header-based).
#   - Radware Bot Manager: served with HTTP 200 (e.g. by iopscience.org/IOP), so it must be sniffed from the
#     body rather than the status code, or it would be miscategorized as a normal (if unhelpful) HTML response.
#   - A generic small JS-redirect "challenge" stub (seen on an Invenio-based institutional repository) that
#     also returns 200 with a tiny loading-spinner page before a script decides whether to let the request
#     through; the literal asset path is the only sniffable signature.
_BOT_BODY_SIGNATURES = (b"botmanager_support@radware.com", b"Bot Manager Captcha", b"/fast-challenge/")


class BlockedByBotProtection(RuntimeError):
    """The server answered with a bot challenge (e.g. Cloudflare, Radware). We stop instead of trying to bypass it."""


@dataclass
class DiscoveredRecord:
    source: str
    external_id: str
    title: str
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    xml_url: str | None = None
    license_raw: str | None = None
    license_evidence: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return f"{self.source}:{self.external_id}"


class Source(Protocol):
    name: str

    def discover(self, terms: list[str], limit: int) -> Iterator[DiscoveredRecord]: ...


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)
    return doi or None


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class PoliteClient:
    """GET-only HTTP client with a per-host minimum interval, backoff on 429/5xx and bot-challenge detection."""

    def __init__(
        self,
        *,
        contact_email: str = "",
        min_interval: float = 1.0,
        max_retries: int = 5,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        agent = f"batterygemma/{__version__} (+https://github.com/fredbuildsai/batterygemma"
        agent += f"; mailto:{contact_email})" if contact_email else ")"
        self._client = httpx.Client(
            headers={"User-Agent": agent}, timeout=timeout, follow_redirects=True, transport=transport
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
        host = urlparse(url).netloc
        for attempt in range(self.max_retries + 1):
            self._respect_interval(host)
            response = self._client.get(url, params=params, headers=headers)
            self._last_request[host] = self._clock()

            if response.status_code == 403 and response.headers.get("cf-mitigated") == "challenge":
                raise BlockedByBotProtection(f"{host} returned a Cloudflare challenge for {url}")
            if any(sig in response.content[:4096] for sig in _BOT_BODY_SIGNATURES):
                raise BlockedByBotProtection(f"{host} returned a bot-management challenge page for {url}")
            if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                delay = _retry_after_seconds(response) or min(120.0, self.min_interval * 2 ** (attempt + 1))
                self._sleep(delay)
                continue
            response.raise_for_status()
            return response
        raise AssertionError("unreachable")  # the final attempt either returns or raises

    def close(self) -> None:
        self._client.close()

    def _respect_interval(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
