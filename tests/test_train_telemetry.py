from batterygemma.train.telemetry import system_power_mw


def test_system_power_mw_uses_system_power_in_while_on_ac(monkeypatch):
    import batterygemma.train.telemetry as telemetry

    plist = {
        "ExternalConnected": True,
        "PowerTelemetryData": {"SystemPowerIn": 12533},
        "InstantAmperage": 0,
        "Voltage": 12159,
    }
    monkeypatch.setattr(telemetry.plistlib, "loads", lambda raw: [plist])
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": b""})())

    assert system_power_mw() == 12533.0


def test_system_power_mw_falls_back_to_amperage_times_voltage_on_battery(monkeypatch):
    import batterygemma.train.telemetry as telemetry

    plist = {
        "ExternalConnected": False,
        "PowerTelemetryData": {},
        "InstantAmperage": 1000,  # 1 A discharge
        "Voltage": 12000,  # 12 V
    }
    monkeypatch.setattr(telemetry.plistlib, "loads", lambda raw: [plist])
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": b""})())

    # 1000 mA * 12000 mV / 1000 = 12000 mW
    assert system_power_mw() == 12000.0


def test_system_power_mw_unwraps_64_bit_signed_amperage(monkeypatch):
    """A discharging amperage sometimes comes back as an unsigned 64-bit wraparound of a negative
    value (confirmed live in a real ioreg dump, e.g. AccumulatedBatteryDischarge=18446744073709551021
    representing -595) - system_power_mw must recover the real magnitude, not treat it as ~1.8e19 mA."""
    import batterygemma.train.telemetry as telemetry

    wrapped = (1 << 64) - 500  # represents -500 mA
    plist = {
        "ExternalConnected": False,
        "PowerTelemetryData": {},
        "InstantAmperage": wrapped,
        "Voltage": 12000,
    }
    monkeypatch.setattr(telemetry.plistlib, "loads", lambda raw: [plist])
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": b""})())

    assert system_power_mw() == 500 * 12000 / 1000.0


def test_system_power_mw_returns_none_when_no_battery_present(monkeypatch):
    import batterygemma.train.telemetry as telemetry

    monkeypatch.setattr(telemetry.plistlib, "loads", lambda raw: [])
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": b""})())

    assert system_power_mw() is None


def test_system_power_mw_returns_none_on_any_failure(monkeypatch):
    import batterygemma.train.telemetry as telemetry

    def _raise(*a, **k):
        raise OSError("ioreg not found")

    monkeypatch.setattr(telemetry.subprocess, "run", _raise)

    assert system_power_mw() is None
