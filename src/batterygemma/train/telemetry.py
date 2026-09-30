"""No-sudo system/GPU telemetry for a training run - see docs/training.md.

Replaces an earlier `sudo powermetrics`-based approach entirely (removed): CPU/GPU power draw split
by component genuinely requires `powermetrics`, which this macOS version refuses to run at all
without root (confirmed live: `powermetrics must be invoked as the superuser`, even for its lightest
sampler) - not worth a sudo password prompt on every training run for that split alone. Everything
below needs no elevated privileges:

- `psutil` (system-wide CPU/memory/swap/battery) - confirmed live that `sensors_temperatures` and
  `sensors_fans` are simply not implemented on macOS (`AttributeError`), so no temperature is
  available through it either; CPU/memory/swap/battery figures are real and exact.
- `mlx.core` (already installed - it's the training backend itself): live GPU/unified-memory figures
  from the actual Metal allocator, plus static device info (name, recommended working-set ceiling).
- `ioreg -rn AppleSmartBattery` (a plain user command, no sudo): Apple's own charger/battery
  telemetry reports real *whole-system* power draw in milliwatts (`PowerTelemetryData.SystemPowerIn`
  while on AC, or amperage*voltage while discharging) - not split by CPU vs GPU vs ANE the way
  `powermetrics` can, but a real wattage figure a sudo-free approach otherwise has no access to at
  all. Byte-to-GiB conversions throughout use 1024, not 1000 - GiB (binary), not decimal GB.
"""

import plistlib
import subprocess
from typing import Any


def system_power_mw() -> float | None:
    """Real, whole-system power draw in mW, via Apple's own charger/battery telemetry - no sudo.
    Returns None on a machine with no battery (e.g. a desktop Mac) or if the read fails for any
    reason: this is a nice-to-have metric, never worth failing a training run over."""
    try:
        raw = subprocess.run(["ioreg", "-rn", "AppleSmartBattery", "-a"],
                              capture_output=True, timeout=5, check=True).stdout
        data = plistlib.loads(raw)
        info = data[0] if isinstance(data, list) else data
        if not info:
            return None
        telemetry = info.get("PowerTelemetryData", {})
        if info.get("ExternalConnected") and telemetry.get("SystemPowerIn") is not None:
            return float(telemetry["SystemPowerIn"])
        amperage = info.get("InstantAmperage")
        voltage = info.get("Voltage")
        if amperage is not None and voltage is not None:
            # amperage is a large unsigned wraparound value while charging/idle on some firmwares;
            # only meaningful as a discharge rate (negative-going) while genuinely on battery.
            signed_amperage = amperage - (1 << 64) if amperage > (1 << 63) else amperage
            return abs(signed_amperage) * voltage / 1000.0
        return None
    except Exception:
        return None


def gpu_memory_gib() -> dict[str, float]:
    """Live Metal/unified-memory figures from MLX's own allocator (GiB, 1024-based)."""
    import mlx.core as mx

    return {
        "gpu_active_memory_gib": round(mx.get_active_memory() / 1024**3, 4),
        "gpu_cache_memory_gib": round(mx.get_cache_memory() / 1024**3, 4),
    }


def device_snapshot() -> dict[str, Any]:
    """Static per-machine info (log once per run, not per step) - `mx.device_info()` plus total
    system RAM via `psutil`, both GiB (1024-based)."""
    import mlx.core as mx
    import psutil

    info = mx.device_info()
    return {
        "device_name": info.get("device_name"),
        "gpu_architecture": info.get("architecture"),
        "gpu_max_recommended_working_set_gib": round(info["max_recommended_working_set_size"] / 1024**3, 3)
        if info.get("max_recommended_working_set_size") else None,
        "system_total_memory_gib": round(psutil.virtual_memory().total / 1024**3, 3),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
    }


def system_snapshot() -> dict[str, Any]:
    """One flat dict of live, no-sudo system + GPU telemetry, ready to merge into a per-step log row."""
    import psutil

    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    battery = psutil.sensors_battery()
    snapshot: dict[str, Any] = {
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_memory_percent": vmem.percent,
        "system_memory_available_gib": round(vmem.available / 1024**3, 3),
        # `free` is near-meaningless alone on macOS (it aggressively repurposes idle memory as disk
        # cache rather than leaving it reported as free) - `available` above is the figure that
        # actually reflects reclaimable memory; `free` is kept here only for completeness/analysis.
        "system_memory_free_gib": round(vmem.free / 1024**3, 3),
        "system_memory_wired_gib": round(getattr(vmem, "wired", 0) / 1024**3, 3),
        "system_swap_percent": swap.percent,
        "system_swap_used_gib": round(swap.used / 1024**3, 3),
        "battery_percent": battery.percent if battery else None,
        "power_plugged": battery.power_plugged if battery else None,
        "system_power_draw_mw": system_power_mw(),
    }
    snapshot.update(gpu_memory_gib())
    return snapshot
