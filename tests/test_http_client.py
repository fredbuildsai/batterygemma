import httpx
import pytest

from batterygemma.sources.base import BlockedByBotProtection, PoliteClient, normalize_doi, normalize_title


def make_client(handler, **kwargs):
    sleeps = []
    client = PoliteClient(transport=httpx.MockTransport(handler), sleep=sleeps.append, min_interval=1.0, **kwargs)
    return client, sleeps


def test_retries_429_honouring_retry_after_then_succeeds():
    responses = iter([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json={"ok": True})])
    client, sleeps = make_client(lambda request: next(responses))
    assert client.get("https://api.example.org/works").json() == {"ok": True}
    assert 7.0 in sleeps


def test_gives_up_after_max_retries():
    client, _ = make_client(lambda request: httpx.Response(503), max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://api.example.org/works")


def test_cloudflare_challenge_raises_instead_of_retrying():
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(403, headers={"cf-mitigated": "challenge", "server": "cloudflare"})

    client, _ = make_client(handler)
    with pytest.raises(BlockedByBotProtection):
        client.get("https://chemrxiv.org/doi/pdf/10.26434/x")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        # Real response body observed from iopscience.iop.org: served with HTTP 200, so it must be sniffed
        # from content rather than relying on status code or a Cloudflare-specific header.
        b'<head><title>Radware Bot Manager Captcha</title><script>...ssk=botmanager_support@radware.com...',
        # Real response body observed from an Invenio-based institutional repository (also HTTP 200): a tiny
        # loading-spinner stub that decides client-side whether to let the request through.
        b'<html><head><link href="/fast-challenge/style.css" rel="stylesheet"/>'
        b'<script src="/fast-challenge/index.js"></script></head><body><div class="loading-overlay">'
        b'<div class="spinner"></div></div></body></html>',
    ],
)
def test_non_cloudflare_bot_challenges_are_detected_despite_status_200(body):
    client, _ = make_client(lambda request: httpx.Response(200, content=body))
    with pytest.raises(BlockedByBotProtection):
        client.get("https://iopscience.iop.org/article/x/pdf")


def test_normal_html_response_is_not_mistaken_for_a_challenge():
    client, _ = make_client(lambda request: httpx.Response(200, content=b"<html><body>A real article page.</body></html>"))
    assert client.get("https://example.org/article").status_code == 200


def test_user_agent_includes_contact_only_when_configured():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200)

    make_client(handler)[0].get("https://api.example.org/")
    assert "mailto" not in seen["ua"]
    make_client(handler, contact_email="team@example.org")[0].get("https://api.example.org/")
    assert seen["ua"].endswith("mailto:team@example.org)")


def test_normalizers():
    assert normalize_doi("https://doi.org/10.26434/ChemRxiv-2025-RVP45") == "10.26434/chemrxiv-2025-rvp45"
    assert normalize_title("A Reflection on Lithium-Ion Battery  Cathode Chemistry!") == (
        "a reflection on lithium ion battery cathode chemistry"
    )
