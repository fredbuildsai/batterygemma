"""Stage 11 (Eval): aggregate per-item `run_eval` results into accuracy metrics with confidence intervals.

Wilson score intervals (not the naive normal approximation) are used because gold categories can be small
(e.g. 200 ideation items), where the normal approximation's interval can fall outside [0, 1].
"""

import math
from typing import Any


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval (default z=1.96 -> 95%) for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo, hi = (centre - spread) / denom, (centre + spread) / denom
    return (max(0.0, lo), min(1.0, hi))


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-category summary: accuracy (+ Wilson 95% CI) where results carry `correct`, plus mean score(s)."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for category, rows in by_category.items():
        n = len(rows)
        entry: dict[str, Any] = {"n": n}
        if "correct" in rows[0]:
            successes = sum(1 for r in rows if r["correct"])
            lo, hi = wilson_interval(successes, n)
            entry["accuracy"] = round(successes / n, 4)
            entry["ci95"] = (round(lo, 4), round(hi, 4))
        if category == "ideation":
            dims = rows[0]["scores"].keys()
            entry["mean_scores"] = {dim: round(sum(r["scores"][dim] for r in rows) / n, 3) for dim in dims}
        elif "score" in rows[0]:
            entry["mean_score"] = round(sum(r["score"] for r in rows) / n, 3)
        summary[category] = entry
    return summary
