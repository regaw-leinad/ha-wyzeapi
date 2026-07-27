"""Tests for the configurable Lock Bolt scan interval and the options merge fix.

Bolt state is poll-only, so the interval is the only thing controlling how quickly a
manual thumbturn or keypad entry is noticed, and the right value depends on how much
BLE traffic an owner is willing to spend on a battery-powered lock. It was hardcoded.

Making it an option exposed a separate pre-existing bug: ``async_setup_entry`` built a
fresh options dict on every startup, so any option other than the bulb toggle was
silently dropped each time Home Assistant restarted.
"""

from unittest.mock import MagicMock, patch

import pytest
import voluptuous as vol

from custom_components.wyzeapi.const import (
    BOLT_SCAN_INTERVAL,
    BULB_LOCAL_CONTROL,
    DEFAULT_BOLT_SCAN_INTERVAL,
    DEFAULT_LOCAL_CONTROL,
    MAX_BOLT_SCAN_INTERVAL,
    MIN_BOLT_SCAN_INTERVAL,
)

LOCK_UUID = "YD_BT1.b38ae5ebd9864864273db2c08fab2c28"


def make_coordinator(scan_interval=None):
    lock = MagicMock()
    lock.mac = LOCK_UUID
    lock.nickname = "Test Bolt"

    with patch("homeassistant.helpers.frame.report_usage"):
        from custom_components.wyzeapi.coordinator import WyzeLockBoltCoordinator

        if scan_interval is None:
            return WyzeLockBoltCoordinator(MagicMock(), MagicMock(), lock)
        return WyzeLockBoltCoordinator(MagicMock(), MagicMock(), lock, scan_interval)


def test_default_interval_matches_previous_hardcoded_value():
    """Existing installs must not silently change behaviour on upgrade."""
    assert DEFAULT_BOLT_SCAN_INTERVAL == 300
    assert make_coordinator().update_interval.total_seconds() == 300


@pytest.mark.parametrize("seconds", [30, 60, 120, 300, 1800, 3600])
def test_interval_is_honoured(seconds):
    assert make_coordinator(seconds).update_interval.total_seconds() == seconds


def test_options_merge_preserves_other_options():
    """The regression: a fresh dict wiped every option except the bulb toggle.

    Reproduces the exact expression from async_setup_entry rather than booting Home
    Assistant, so it pins the merge semantics.
    """
    existing = {BOLT_SCAN_INTERVAL: 120, "some_future_option": "keep me"}

    merged = {
        **existing,
        BULB_LOCAL_CONTROL: existing.get(BULB_LOCAL_CONTROL, DEFAULT_LOCAL_CONTROL),
    }

    assert merged[BOLT_SCAN_INTERVAL] == 120, "interval must survive a restart"
    assert merged["some_future_option"] == "keep me"
    assert merged[BULB_LOCAL_CONTROL] == DEFAULT_LOCAL_CONTROL


def test_options_merge_does_not_clobber_an_explicit_bulb_setting():
    existing = {BULB_LOCAL_CONTROL: False, BOLT_SCAN_INTERVAL: 600}

    merged = {
        **existing,
        BULB_LOCAL_CONTROL: existing.get(BULB_LOCAL_CONTROL, DEFAULT_LOCAL_CONTROL),
    }

    assert merged[BULB_LOCAL_CONTROL] is False
    assert merged[BOLT_SCAN_INTERVAL] == 600


def test_old_replace_semantics_would_have_dropped_the_interval():
    """Guards the intent: shows the pre-fix expression losing data."""
    existing = {BOLT_SCAN_INTERVAL: 120}

    old_style = {
        BULB_LOCAL_CONTROL: existing.get(BULB_LOCAL_CONTROL, DEFAULT_LOCAL_CONTROL)
    }

    assert BOLT_SCAN_INTERVAL not in old_style


def _schema():
    return vol.Schema(
        {
            vol.Optional(
                BOLT_SCAN_INTERVAL, default=DEFAULT_BOLT_SCAN_INTERVAL
            ): vol.All(
                vol.Coerce(int),
                vol.Range(min=MIN_BOLT_SCAN_INTERVAL, max=MAX_BOLT_SCAN_INTERVAL),
            )
        }
    )


@pytest.mark.parametrize("value", [MIN_BOLT_SCAN_INTERVAL, 300, MAX_BOLT_SCAN_INTERVAL])
def test_schema_accepts_values_in_range(value):
    assert _schema()({BOLT_SCAN_INTERVAL: value})[BOLT_SCAN_INTERVAL] == value


@pytest.mark.parametrize("value", [0, -1, 29, 3601, 86400])
def test_schema_rejects_values_out_of_range(value):
    """A 0 or negative interval would busy-loop the lock's radio flat."""
    with pytest.raises(vol.Invalid):
        _schema()({BOLT_SCAN_INTERVAL: value})


def test_schema_coerces_string_input():
    """Options flow submits form values as strings."""
    assert _schema()({BOLT_SCAN_INTERVAL: "600"})[BOLT_SCAN_INTERVAL] == 600


def test_lower_bound_leaves_room_for_a_cold_connect():
    """Each poll costs a 2-7s connect, so the floor must not overlap the next poll."""
    assert MIN_BOLT_SCAN_INTERVAL > 7
