"""Timed runs happen on mains power only.

A laptop caps CPU and GPU power when unplugged, so a number measured on battery
describes the battery. `run_on_mains` waits while the machine is on battery,
then runs the timed function with a watcher that samples the power state; if
the charger went at any point during the run, the result is thrown away and
the run repeated once power is back. Machines that report no power supplies
(desktops, VMs) count as mains.

MOJO_DLLM_POWER_SUPPLY points the check at another directory laid out like
/sys/class/power_supply, so a test can unplug a fake charger mid-run.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import threading
import time
from typing import Callable, TypeVar

POWER_SUPPLY = pathlib.Path(os.environ.get("MOJO_DLLM_POWER_SUPPLY", "/sys/class/power_supply"))
T = TypeVar("T")


def on_battery(root: pathlib.Path = POWER_SUPPLY) -> bool | None:
    """True when no mains or USB supply is online and a battery discharges;
    None when the machine reports no supplies at all."""
    if not root.is_dir():
        return None

    def read(d: pathlib.Path, name: str) -> str:
        try:
            return (d / name).read_text().strip()
        except OSError:
            return ""

    supplies = [d for d in root.iterdir() if d.is_dir()]
    if any(read(d, "type") in ("Mains", "USB") and read(d, "online") == "1" for d in supplies):
        return False
    if any(read(d, "type") == "Battery" and read(d, "status") == "Discharging" for d in supplies):
        return True
    return None


def wait_for_mains(root: pathlib.Path = POWER_SUPPLY, *, poll_s: float = 5.0, allow_battery: bool = False,
                   sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
                   on_pause: Callable[[], None] | None = None,
                   on_resume: Callable[[], None] | None = None) -> float:
    """Block while on battery and return the seconds waited (0 on mains)."""
    if allow_battery or not on_battery(root):
        return 0
    log(f"on battery since {dt.datetime.now():%H:%M:%S}: paused, plug in the charger to continue")
    if on_pause:
        on_pause()
    waited = 0.0
    while on_battery(root):
        sleep(poll_s)
        waited += poll_s
    log(f"charger back after {waited:.0f} s: resuming")
    if on_resume:
        on_resume()
    return waited


class _Watch:
    """Samples the power state every `sample_s` seconds until stopped."""

    def __init__(self, root: pathlib.Path, sample_s: float) -> None:
        self.root, self.sample_s = root, sample_s
        self.saw_battery = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if on_battery(self.root):
                self.saw_battery = True
            self._stop.wait(self.sample_s)

    def __enter__(self) -> "_Watch":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        if on_battery(self.root):   # the last moment, which a sample may have missed
            self.saw_battery = True


def run_on_mains(fn: Callable[[], T], *, root: pathlib.Path = POWER_SUPPLY, allow_battery: bool = False,
                 poll_s: float = 5.0, sample_s: float = 1.0, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print, on_pause: Callable[[], None] | None = None,
                 on_resume: Callable[[], None] | None = None) -> tuple[T, dict]:
    """Run `fn` on mains power, repeating it when the charger went mid-run.

    Returns fn's result and what it took: seconds paused, runs thrown away,
    and whether the kept run was on battery (only possible with allow_battery).
    """
    if allow_battery:
        return fn(), {"power_waited_s": 0, "power_retries": 0, "on_battery": bool(on_battery(root))}
    waited, retries = 0.0, 0
    while True:
        waited += wait_for_mains(root, poll_s=poll_s, sleep=sleep, log=log, on_pause=on_pause, on_resume=on_resume)
        with _Watch(root, sample_s) as watch:
            result = fn()
        if not watch.saw_battery:
            return result, {"power_waited_s": waited, "power_retries": retries, "on_battery": False}
        retries += 1
        log("the charger went during this run: its result is discarded, and it will repeat on mains")
