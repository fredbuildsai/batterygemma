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
