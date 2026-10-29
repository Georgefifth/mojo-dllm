"""bench/power.py: timed runs happen on mains power only. A laptop caps CPU and
GPU power when unplugged, so a run that saw the battery is paused before it
starts, or thrown away and repeated if the charger went during it."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import power  # noqa: E402


def _supply(root: Path, name: str, kind: str, **files: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "type").write_text(kind + "\n")
    for k, v in files.items():
        (d / k).write_text(v + "\n")


def _laptop(root: Path, plugged: bool) -> None:
    _supply(root, "ADP1", "Mains", online="1" if plugged else "0")
    _supply(root, "BAT1", "Battery", status="Charging" if plugged else "Discharging")


# on_battery


def test_unplugged_laptop_is_on_battery(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=False)
    _supply(tmp_path, "ucsi-source-psy-USBC000:001", "USB", online="0")
    assert power.on_battery(tmp_path) is True


def test_mains_online_is_not_battery(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=True)
    assert power.on_battery(tmp_path) is False


def test_usb_c_charger_counts_as_plugged_in(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=False)
    _supply(tmp_path, "ucsi-source-psy-USBC000:001", "USB", online="1")
    assert power.on_battery(tmp_path) is False


def test_full_battery_on_mains_is_not_battery(tmp_path: Path) -> None:
    _supply(tmp_path, "ADP1", "Mains", online="1")
    _supply(tmp_path, "BAT1", "Battery", status="Full")
    assert power.on_battery(tmp_path) is False


def test_desktop_without_supplies_is_unknown(tmp_path: Path) -> None:
    assert power.on_battery(tmp_path / "missing") is None
    assert power.on_battery(tmp_path) is None


# wait_for_mains


def test_wait_returns_at_once_on_mains(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=True)
    logs: list[str] = []
    assert power.wait_for_mains(tmp_path, sleep=pytest.fail, log=logs.append) == 0
    assert logs == []


def test_wait_pauses_until_the_charger_returns(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=False)
    naps: list[float] = []
    logs: list[str] = []
    events: list[str] = []

    def sleep(s: float) -> None:
        naps.append(s)
        if len(naps) == 3:
            _laptop(tmp_path, plugged=True)

    waited = power.wait_for_mains(tmp_path, poll_s=5, sleep=sleep, log=logs.append,
                                  on_pause=lambda: events.append("pause"),
                                  on_resume=lambda: events.append("resume"))
    assert naps == [5, 5, 5]
    assert waited == 15
    assert events == ["pause", "resume"]
    assert len(logs) == 2 and "plug in" in logs[0] and "resum" in logs[1]


def test_wait_is_skipped_when_battery_is_allowed(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=False)
    assert power.wait_for_mains(tmp_path, allow_battery=True, sleep=pytest.fail, log=pytest.fail) == 0


# run_on_mains


def test_a_run_on_mains_happens_once(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=True)
    calls: list[int] = []
    result, info = power.run_on_mains(lambda: calls.append(1) or "ok", root=tmp_path, sample_s=0.01, log=pytest.fail)
    assert result == "ok" and calls == [1]
    assert info == {"power_waited_s": 0, "power_retries": 0, "on_battery": False}


def test_unplugging_during_a_run_repeats_it(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=True)
    calls: list[int] = []
    logs: list[str] = []

    def run() -> int:
        calls.append(1)
        if len(calls) == 1:
            _laptop(tmp_path, plugged=False)  # the charger goes mid-run ...

        return len(calls)

    def sleep(s: float) -> None:
        _laptop(tmp_path, plugged=True)       # ... and comes back while paused

    result, info = power.run_on_mains(run, root=tmp_path, sample_s=0.01, sleep=sleep, log=logs.append)
    assert result == 2 and len(calls) == 2
    assert info["power_retries"] == 1 and info["on_battery"] is False
    assert any("repeat" in m for m in logs)


def test_a_short_unplug_inside_a_run_is_caught(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=True)
    calls: list[int] = []

    def run() -> None:
        calls.append(1)
        if len(calls) == 1:
            _laptop(tmp_path, plugged=False)
            time.sleep(0.3)                    # the watcher samples every 10 ms
            _laptop(tmp_path, plugged=True)    # plugged back before the run ends

    _, info = power.run_on_mains(run, root=tmp_path, sample_s=0.01, log=lambda m: None)
    assert len(calls) == 2 and info["power_retries"] == 1


def test_battery_allowed_runs_once_and_records_it(tmp_path: Path) -> None:
    _laptop(tmp_path, plugged=False)
    calls: list[int] = []
    _, info = power.run_on_mains(lambda: calls.append(1), root=tmp_path, allow_battery=True,
                                 sample_s=0.01, sleep=pytest.fail, log=pytest.fail)
    assert calls == [1]
    assert info == {"power_waited_s": 0, "power_retries": 0, "on_battery": True}
