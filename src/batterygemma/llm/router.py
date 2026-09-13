"""Quota-aware teacher-LLM router with automatic failover.

Each task route lists deployments in preference order. For every call the router walks that list and
skips deployments that are disabled (no API key), paid while paid use is off or over budget, in an
excluded model family (judge != generator), cooling down after a 429/error, or over their per-minute or
per-day request limits. Every attempt is recorded in `llm_calls`, which doubles as the daily quota
ledger and as a response cache keyed by prompt hash.
"""

import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, func, select

from batterygemma.db.models import LLMCall
from batterygemma.db.session import get_session

Message = dict[str, Any]
Validator = Callable[[str], Any]


class AllDeploymentsExhausted(RuntimeError):
    """No deployment on the route could produce a valid response."""


@dataclass
class Deployment:
    name: str
    model: str
    api_key_env: str
    tier: str = "free"
    family: str = ""
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tags: list[str] = field(default_factory=list)
    cost_per_mtok_in: float = 0.0
    cost_per_mtok_out: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.cost_per_mtok_in + tokens_out * self.cost_per_mtok_out) / 1_000_000


@dataclass
class LLMResult:
    text: str
    parsed: Any
    deployment: str
    model: str
    family: str
    cached: bool = False


def _utc_midnight() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _classify_error(exc: Exception) -> str:
    """Map provider exceptions (LiteLLM wraps them with status codes) to router outcomes."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if status in (401, 403) or "authentication" in name or "permissiondenied" in name:
        return "auth"
    if status == 429 or "ratelimit" in name:
        daily_markers = ("per day", "daily", "quota", "requests per day", "rpd", "exceeded your current")
        return "quota" if any(m in text for m in daily_markers) else "rate_limited"
    return "error"


class LLMRouter:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        engine: Engine | None = None,
        allow_paid: bool = False,
        max_usd_per_day: float = 0.0,
        completion_fn: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.deployments = {d["name"]: Deployment(**d) for d in config["deployments"]}
        self.routes: dict[str, list[str]] = config["routes"]
        unknown = {n for chain in self.routes.values() for n in chain} - self.deployments.keys()
        if unknown:
            raise ValueError(f"Routes reference unknown deployments: {sorted(unknown)}")
        cooldown = config.get("cooldown", {})
        self.cooldown_rate = cooldown.get("rate_limit_seconds", 60)
        self.cooldown_quota = cooldown.get("daily_quota_seconds", 86400)
        self.cooldown_error = cooldown.get("error_seconds", 30)
        self.max_attempts = config.get("max_attempts_per_call", 6)
        self.engine = engine
        self.allow_paid = allow_paid
        self.max_usd_per_day = max_usd_per_day
        self._completion_fn = completion_fn
        self._clock = clock
        self._sleep = sleep
        self._cooldown_until: dict[str, float] = {}
        self._disabled: set[str] = set()
        self._recent: dict[str, deque[float]] = {}

    # --- public API ------------------------------------------------------------------------------

    def complete(
        self,
        route: str,
        messages: Sequence[Message],
        *,
        exclude_families: Sequence[str] = (),
        validate: Validator | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> LLMResult:
        """Return the first valid response along the route, failing over as needed.

        `validate` receives the raw text and returns the parsed value or raises ValueError, in which
        case the attempt is logged as invalid_output and the next deployment is tried.
        """
        if route not in self.routes:
            raise KeyError(f"Unknown route: {route}")
        params = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode}
        prompt_hash = self._hash(route, messages, params)

        if use_cache and (cached := self._cached(prompt_hash, exclude_families, validate)):
            return cached

        attempts = 0
        tried: set[str] = set()
        while attempts < self.max_attempts:
            deployment, wait = self._next_deployment(route, exclude_families, tried)
            if deployment is None:
                if wait is None:
                    break
                self._sleep(wait)
                continue
            attempts += 1
            result = self._attempt(route, deployment, messages, params, prompt_hash, validate)
            if result is not None:
                return result
            tried.add(deployment.name)
        raise AllDeploymentsExhausted(
            f"Route '{route}' exhausted after {attempts} attempts (excluded families: {list(exclude_families)})"
        )

    def status(self) -> list[dict[str, Any]]:
        used = self._requests_today()
        now = self._clock()
        rows = []
        for d in self.deployments.values():
            rows.append(
                {
                    "name": d.name,
                    "model": d.model,
                    "tier": d.tier,
                    "family": d.family,
                    "enabled": d.enabled and d.name not in self._disabled,
                    "used_today": used.get(d.name, 0),
                    "rpd": d.rpd,
                    "cooldown_s": max(0, int(self._cooldown_until.get(d.name, 0) - now)),
                }
            )
        return rows

    def spent_today_usd(self) -> float:
        with get_session(self.engine) as s:
            total = s.scalar(
                select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                    LLMCall.tier == "paid", LLMCall.created_at >= _utc_midnight()
                )
            )
        return float(total or 0.0)

    # --- selection -------------------------------------------------------------------------------

    def _next_deployment(
        self, route: str, exclude_families: Sequence[str], tried: set[str]
    ) -> tuple[Deployment | None, float | None]:
        """Pick the first usable deployment; if only per-minute limits block, return how long to wait."""
        now = self._clock()
        used_today = self._requests_today()
        min_wait: float | None = None
        for name in self.routes[route]:
            d = self.deployments[name]
            if name in tried or name in self._disabled or not d.enabled:
                continue
            if d.family and d.family in exclude_families:
                continue
            if d.tier == "paid" and (not self.allow_paid or self.spent_today_usd() >= self.max_usd_per_day):
                continue
            if self._cooldown_until.get(name, 0) > now:
                continue
            if d.rpd is not None and used_today.get(name, 0) >= d.rpd:
                continue
            if d.rpm is not None:
                window = self._recent.setdefault(name, deque())
                while window and now - window[0] >= 60:
                    window.popleft()
                if len(window) >= d.rpm:
                    wait = 60 - (now - window[0])
                    min_wait = wait if min_wait is None else min(min_wait, wait)
                    continue
            return d, None
        return None, min_wait

    def _requests_today(self) -> dict[str, int]:
        with get_session(self.engine) as s:
            rows = s.execute(
                select(LLMCall.deployment, func.count())
                .where(LLMCall.created_at >= _utc_midnight(), LLMCall.status != "cache_hit")
                .group_by(LLMCall.deployment)
            ).all()
        return {name: count for name, count in rows}

    # --- execution -------------------------------------------------------------------------------

    def _attempt(
        self,
        route: str,
        d: Deployment,
        messages: Sequence[Message],
        params: dict[str, Any],
        prompt_hash: str,
        validate: Validator | None,
    ) -> LLMResult | None:
        self._recent.setdefault(d.name, deque()).append(self._clock())
        kwargs: dict[str, Any] = {
            "model": d.model,
            "messages": list(messages),
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "api_key": os.environ.get(d.api_key_env),
            "timeout": 180,
        }
        if params["json_mode"]:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = self._completion()(**kwargs)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            outcome = _classify_error(exc)
            latency = int((time.perf_counter() - started) * 1000)
            if outcome == "auth":
                self._disabled.add(d.name)
            else:
                seconds = {"quota": self.cooldown_quota, "rate_limited": self.cooldown_rate}.get(
                    outcome, self.cooldown_error
                )
                self._cooldown_until[d.name] = self._clock() + seconds
            self._log(route, d, prompt_hash, status=outcome, error=str(exc)[:2000], latency_ms=latency)
            return None

        latency = int((time.perf_counter() - started) * 1000)
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        cost = d.cost(tokens_in, tokens_out) if d.tier == "paid" else 0.0

        try:
            parsed = validate(text) if validate else text
        except ValueError as exc:
            self._log(
                route, d, prompt_hash, status="invalid_output", error=str(exc)[:2000], latency_ms=latency,
                tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
            )
            return None

        self._log(
            route, d, prompt_hash, status="ok", latency_ms=latency, tokens_in=tokens_in,
            tokens_out=tokens_out, cost_usd=cost, response_text=text,
        )
        return LLMResult(text=text, parsed=parsed, deployment=d.name, model=d.model, family=d.family)

    def _cached(
        self, prompt_hash: str, exclude_families: Sequence[str], validate: Validator | None
    ) -> LLMResult | None:
        with get_session(self.engine) as s:
            rows = s.scalars(
                select(LLMCall)
                .where(LLMCall.prompt_hash == prompt_hash, LLMCall.status == "ok")
                .order_by(LLMCall.created_at.desc())
            ).all()
        for row in rows:
            d = self.deployments.get(row.deployment)
            family = d.family if d else ""
            if family and family in exclude_families:
                continue
            try:
                parsed = validate(row.response_text or "") if validate else row.response_text
            except ValueError:
                continue
            return LLMResult(
                text=row.response_text or "", parsed=parsed, deployment=row.deployment, model=row.model,
                family=family, cached=True,
            )
        return None

    def _completion(self) -> Callable[..., Any]:
        if self._completion_fn is None:
            import litellm  # imported lazily: slow import, not needed for tests or status

            litellm.suppress_debug_info = True
            self._completion_fn = litellm.completion
        return self._completion_fn

    def _log(self, route: str, d: Deployment, prompt_hash: str, *, status: str, **fields: Any) -> None:
        with get_session(self.engine) as s:
            s.add(LLMCall(route=route, deployment=d.name, model=d.model, tier=d.tier, prompt_hash=prompt_hash,
                          status=status, **fields))

    @staticmethod
    def _hash(route: str, messages: Sequence[Message], params: dict[str, Any]) -> str:
        payload = json.dumps({"route": route, "messages": list(messages), "params": params}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
