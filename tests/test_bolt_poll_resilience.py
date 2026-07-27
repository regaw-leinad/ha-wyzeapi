"""Tests for Wyze Lock Bolt poll resilience.

The Bolt accepts only one BLE connection at a time, so whenever the owner's phone app
is talking to the lock every HA poll fails. Treating that as a hard failure marks the
lock ``unavailable`` for a whole heartbeat interval, which is worse than serving a
state that is a couple of minutes stale. These tests pin the recovery behaviour:

* a recoverable failure keeps the last known state and schedules an earlier retry
* the scheduled retry is safe to run on the event loop
* a run of consecutive failures still surfaces as ``UpdateFailed``
* a success resets the counter, so intermittent contention never accumulates
* with no previous state there is nothing to serve, so it fails immediately
* a missing characteristic is never treated as transient
"""

from unittest.mock import AsyncMock, MagicMock, patch

from bleak.exc import BleakCharacteristicNotFoundError, BleakError
import pytest

from homeassistant.core import HassJob, HassJobType
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.wyzeapi.const import (
    BOLT_MAX_TRANSIENT_FAILURES,
    BOLT_TRANSIENT_RETRY_SECONDS,
)

LOCK_UUID = "YD_BT1.b38ae5ebd9864864273db2c08fab2c28"


def make_coordinator():
    lock = MagicMock()
    lock.mac = LOCK_UUID
    lock.nickname = "Test Bolt"

    with patch("homeassistant.helpers.frame.report_usage"):
        from custom_components.wyzeapi.coordinator import WyzeLockBoltCoordinator

        coordinator = WyzeLockBoltCoordinator(MagicMock(), MagicMock(), lock)

    coordinator._mac = "28:68:47:E9:D0:7B"
    coordinator._disconnect = AsyncMock()
    coordinator.async_update_listeners = MagicMock()
    return coordinator


def with_known_state(coordinator, state=1):
    """Give the coordinator a previous reading to fall back on."""
    coordinator.data = {"state": state, "timestamp": None}
    return coordinator.data


@pytest.mark.asyncio
async def test_transient_failure_keeps_last_known_state():
    coordinator = make_coordinator()
    known = with_known_state(coordinator)
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("no slot free"))

    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        result = await coordinator._async_update_data()

    assert result == known, "must serve the last reading rather than go unavailable"


@pytest.mark.asyncio
async def test_transient_failure_schedules_an_earlier_retry():
    """Retrying sooner than the heartbeat is the point: contention is short-lived."""
    coordinator = make_coordinator()
    with_known_state(coordinator)
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("busy"))

    with patch("custom_components.wyzeapi.coordinator.async_call_later") as later:
        await coordinator._async_update_data()

    assert later.call_count == 1
    assert later.call_args[0][1] == BOLT_TRANSIENT_RETRY_SECONDS
    assert (
        BOLT_TRANSIENT_RETRY_SECONDS < coordinator.update_interval.total_seconds()
    ), "a retry that is not sooner than the next poll would be pointless"


@pytest.mark.asyncio
async def test_the_scheduled_retry_runs_on_the_event_loop():
    """A bare callable here is not merely untidy, it is a thread-safety bug.

    ``async_call_later`` classifies its target through ``HassJob``. Anything that is
    neither a coroutine function nor an ``@callback`` is treated as blocking and run in
    an executor thread, so touching ``hass`` from it trips Home Assistant's thread
    safety check and logs a report against this integration on every retry. Caught in
    production with a lambda here, hence the test.
    """
    coordinator = make_coordinator()
    with_known_state(coordinator)
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("busy"))

    with patch("custom_components.wyzeapi.coordinator.async_call_later") as later:
        await coordinator._async_update_data()

    target = later.call_args[0][2]
    assert (
        HassJob(target).job_type is not HassJobType.Executor
    ), "async_call_later would run this off the event loop"

    # ...and it must actually ask for the refresh once awaited.
    coordinator.async_request_refresh = AsyncMock()
    await target(None)
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_device_not_visible_is_treated_as_transient():
    """A lock that no scanner can see right now is the commonest contention symptom."""
    coordinator = make_coordinator()
    known = with_known_state(coordinator)
    coordinator._get_ble_client = AsyncMock(return_value=None)

    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        result = await coordinator._async_update_data()

    assert result == known


@pytest.mark.asyncio
async def test_gives_up_after_repeated_failures():
    coordinator = make_coordinator()
    with_known_state(coordinator)
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("still busy"))

    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        for _ in range(BOLT_MAX_TRANSIENT_FAILURES):
            await coordinator._async_update_data()

        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()


@pytest.mark.asyncio
async def test_success_resets_the_failure_counter():
    """Otherwise occasional contention would eventually trip the limit for no reason."""
    coordinator = make_coordinator()
    with_known_state(coordinator)

    client = MagicMock()
    block = _locked_block()
    client.read_gatt_char = AsyncMock(return_value=block)

    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("busy"))
    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        for _ in range(BOLT_MAX_TRANSIENT_FAILURES):
            await coordinator._async_update_data()
    assert coordinator._transient_failures == BOLT_MAX_TRANSIENT_FAILURES

    coordinator._get_ble_client = AsyncMock(return_value=client)
    await coordinator._async_update_data()
    assert coordinator._transient_failures == 0

    # And the budget is fully restored, not partially.
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("busy again"))
    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        for _ in range(BOLT_MAX_TRANSIENT_FAILURES):
            await coordinator._async_update_data()


@pytest.mark.asyncio
async def test_fails_immediately_when_there_is_no_state_to_serve():
    """Fresh setup: falling back to a state of None would publish a bogus reading."""
    coordinator = make_coordinator()
    coordinator.data = {"state": None, "timestamp": None}
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("out of range"))

    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()


@pytest.mark.asyncio
async def test_missing_characteristic_is_not_transient():
    """Wrong device or firmware trouble must surface, not hide behind a stale value."""
    coordinator = make_coordinator()
    with_known_state(coordinator)

    client = MagicMock()
    client.read_gatt_char = AsyncMock(
        side_effect=BleakCharacteristicNotFoundError("2220")
    )
    coordinator._get_ble_client = AsyncMock(return_value=client)

    with pytest.raises(UpdateFailed):
        await coordinator._async_update_data()


@pytest.mark.asyncio
async def test_link_is_released_even_when_connecting_fails():
    """The one connection slot must be freed on the connect path too, not just reads."""
    coordinator = make_coordinator()
    with_known_state(coordinator)
    coordinator._get_ble_client = AsyncMock(side_effect=BleakError("connect failed"))

    with patch("custom_components.wyzeapi.coordinator.async_call_later"):
        await coordinator._async_update_data()

    coordinator._disconnect.assert_awaited()


@pytest.mark.asyncio
async def test_command_in_flight_skips_the_poll():
    """Pre-existing behaviour: never poll over a running command."""
    coordinator = make_coordinator()
    known = with_known_state(coordinator)
    coordinator._current_command = "lock"
    coordinator._get_ble_client = AsyncMock(
        side_effect=AssertionError("must not connect")
    )

    assert await coordinator._async_update_data() == known


def _locked_block():
    from custom_components.wyzeapi.ydble_utils import encrypt_ecb

    return encrypt_ecb(LOCK_UUID[-16:].lower(), bytes([1, 0, 0, 0, 0]) + b"\x00" * 11)
