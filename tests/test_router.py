import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.db.models import LLMCall
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


def response(text, tokens_in=10, tokens_out=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


CONFIG = {
    "deployments": [
        {"name": "a", "model": "p/a", "api_key_env": "KEY_A", "family": "fam1", "rpd": 100},
        {"name": "b", "model": "p/b", "api_key_env": "KEY_B", "family": "fam2", "rpd": 1},
        {"name": "c", "model": "p/c", "api_key_env": "KEY_C", "family": "fam3", "tier": "paid",
         "cost_per_mtok_in": 1_000_000.0, "cost_per_mtok_out": 0.0},
    ],
    "routes": {"qa": ["a", "b", "c"], "judge": ["a", "b"]},
    "cooldown": {"rate_limit_seconds": 60, "daily_quota_seconds": 86400, "error_seconds": 30},
    "max_attempts_per_call": 5,
}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for key in ("KEY_A", "KEY_B", "KEY_C"):
        monkeypatch.setenv(key, "x")


def make_router(engine, behaviour, **kwargs):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        outcome = behaviour(kw["model"], len(calls))
        if isinstance(outcome, Exception):
            raise outcome
        return response(outcome)

    return LLMRouter(CONFIG, engine=engine, completion_fn=completion, sleep=lambda _: None, **kwargs), calls


def statuses(engine):
    with get_session(engine) as s:
        return [(c.deployment, c.status) for c in s.scalars(select(LLMCall).order_by(LLMCall.id))]


def test_rate_limit_fails_over_to_next_deployment(engine):
    router, calls = make_router(engine, lambda model, n: RateLimitError("slow down") if model == "p/a" else "ok")
    result = router.complete("qa", [{"role": "user", "content": "hi"}])
    assert result.deployment == "b"
    assert calls == ["p/a", "p/b"]
    assert statuses(engine) == [("a", "rate_limited"), ("b", "ok")]
    # 'a' is cooling down and 'b' used its single daily request, so a new prompt has nowhere to go
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "again"}], use_cache=False)
    assert calls == ["p/a", "p/b"]


def test_daily_quota_marker_gets_long_cooldown(engine):
    router, _ = make_router(
        engine, lambda model, n: RateLimitError("Quota exceeded: requests per day") if model == "p/a" else "ok"
    )
    router.complete("qa", [{"role": "user", "content": "hi"}])
    assert statuses(engine)[0] == ("a", "quota")
    assert router.status()[0]["cooldown_s"] > 80000


def test_rpd_limit_skips_exhausted_deployment_and_paid_is_off_by_default(engine):
    router, calls = make_router(engine, lambda model, n: RateLimitError("x") if model == "p/a" else "ok")
    router.complete("qa", [{"role": "user", "content": "first"}])  # uses b's only daily request
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "second"}], use_cache=False)
    assert "p/c" not in calls  # paid deployment never used without allow_paid


def test_paid_budget_is_enforced(engine):
    router, calls = make_router(
        engine, lambda model, n: "ok" if model == "p/c" else RateLimitError("x"), allow_paid=True, max_usd_per_day=5
    )
    router.complete("qa", [{"role": "user", "content": "one"}])  # costs $10 (10 tokens x $1/token)
    assert calls[-1] == "p/c"
    router._cooldown_until.clear()
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "two"}])
    assert router.spent_today_usd() == pytest.approx(10.0)


def test_judge_excludes_generator_family(engine):
    router, calls = make_router(engine, lambda model, n: "ok")
    result = router.complete("judge", [{"role": "user", "content": "grade"}], exclude_families=["fam1"])
    assert result.deployment == "b" and calls == ["p/b"]


def test_invalid_output_retries_on_next_model_and_cache_hits(engine):
    def validate(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("not json") from exc

    router, calls = make_router(engine, lambda model, n: "not json" if model == "p/a" else '{"q": 1}')
    messages = [{"role": "user", "content": "give json"}]
    first = router.complete("qa", messages, validate=validate)
    assert first.parsed == {"q": 1} and first.deployment == "b"
    second = router.complete("qa", messages, validate=validate)
    assert second.cached and second.parsed == {"q": 1}
    assert calls == ["p/a", "p/b"]  # no new provider call for the cached prompt


SINGLE_PROVIDER = {
    "deployments": [
        {"name": "gen", "model": "mistral/mistral-medium-latest", "api_key_env": "KEY_A", "family": "mistral"},
        {"name": "judge", "model": "mistral/magistral-medium-latest", "api_key_env": "KEY_A", "family": "mistral"},
    ],
    "routes": {"judge": ["gen", "judge"]},
}


def test_same_family_fallback_uses_a_different_model_and_flags_the_result(engine):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        return response("5")

    router = LLMRouter(SINGLE_PROVIDER, engine=engine, completion_fn=completion)
    messages = [{"role": "user", "content": "score this"}]
    generator = {"exclude_families": ["mistral"], "exclude_models": ["mistral/mistral-medium-latest"]}

    with pytest.raises(AllDeploymentsExhausted):
        router.complete("judge", messages, **generator)

    result = router.complete("judge", messages, allow_same_family_fallback=True, **generator)
    assert result.model == "mistral/magistral-medium-latest"
    assert result.relaxed_family
    assert calls == ["mistral/magistral-medium-latest"]


class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


WORKSPACE = {
    "deployments": [
        {"name": "small", "model": "mistral/small", "api_key_env": "KEY_A", "family": "mistral", "rate_group": "ws"},
        {"name": "medium", "model": "mistral/medium", "api_key_env": "KEY_A", "family": "mistral", "rate_group": "ws"},
    ],
    "routes": {"qa": ["small", "medium"]},
    "cooldown": {"rate_limit_seconds": 60},
}


def test_rate_group_shares_cooldown_then_waits_and_retries_same_model(engine):
    clock = FakeClock()
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        if len(calls) == 1:
            raise RateLimitError('{"message":"Rate limit exceeded","raw_status_code":429}')
        return response("ok")

    router = LLMRouter(WORKSPACE, engine=engine, completion_fn=completion, clock=clock, sleep=clock.sleep)
    result = router.complete("qa", [{"role": "user", "content": "hi"}], max_wait_seconds=120)

    # 'medium' shares the workspace budget, so it is not hammered after the 429; the router waits out the
    # shared cooldown and retries 'small'.
    assert calls == ["mistral/small", "mistral/small"]
    assert result.deployment == "small"
    assert clock.now == pytest.approx(1_060.0)


def test_rate_group_gives_up_when_wait_budget_is_too_small(engine):
    calls = []

    def always_rate_limited(**kw):
        calls.append(kw["model"])
        raise RateLimitError("slow down")

    router = LLMRouter(WORKSPACE, engine=engine, completion_fn=always_rate_limited, sleep=lambda _: None)
    with pytest.raises(AllDeploymentsExhausted):
        router.complete("qa", [{"role": "user", "content": "hi"}], max_wait_seconds=10)
    assert calls == ["mistral/small"]  # the sibling in the same rate group is never tried


def test_tier_not_allowed_disables_model_and_fails_over(engine):
    class APIError(Exception):
        status_code = 403

    tier_error = APIError('{"message":"This model is not available in your subscription tier","type":"tier_not_allowed"}')
    router, calls = make_router(engine, lambda model, n: tier_error if model == "p/a" else "ok")
    result = router.complete("qa", [{"role": "user", "content": "hi"}])
    assert result.deployment == "b"
    assert statuses(engine)[0] == ("a", "unavailable")
    assert not router.status()[0]["enabled"]


def test_auth_error_disables_deployment(engine):
    router, calls = make_router(engine, lambda model, n: AuthenticationError("bad key") if model == "p/a" else "ok")
    router.complete("qa", [{"role": "user", "content": "hi"}])
    assert not router.status()[0]["enabled"]
