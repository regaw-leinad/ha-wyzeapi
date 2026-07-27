"""Tests for Wyze Lock Bolt command reliability on a reused BLE connection.

The Bolt is driven over a single BLE connection that ``lock_unlock()`` deliberately
keeps alive for a short window after a command, so a follow-up command reuses it
instead of paying another 2-7s cold connect. Two bugs lived in that reuse path:

1. ``start_notify()`` was called unconditionally on every command. On an ESPHome
   Bluetooth proxy, ``bleak_esphome`` raises
   ``BleakError: Notifications are already enabled`` when a handle is subscribed
   twice, because ``aioesphomeapi`` stores the unsubscribe handle in a plain dict
   keyed on ``(address, handle)`` and a second subscribe would orphan the first
   callback. The command then silently never ran.

2. The teardown timer was fire-and-forget, so the previous command's timer fired
   partway through the next command and cleared ``_current_command``, making a
   failed command look like a state that simply reverted.

These tests drive the real ``lock_unlock()`` against a fake client that emulates the
proxy's strict subscription accounting, so they fail if either fix regresses.
"""

import asyncio
from unittest.mock import MagicMock, patch

from bleak.exc import BleakError
import pytest

from custom_components.wyzeapi.const import (
    BOLT_COMMAND_DISCONNECT_SECONDS,
    BOLT_COMMAND_TIMEOUT_SECONDS,
    YDBLE_CON_TYPE_PLAINTEXT,
    YDBLE_CON_TYPE_UUID,
    YDBLE_UART_RX_UUID,
)
from custom_components.wyzeapi.coordinator import NOTIFY_UUIDS
from custom_components.wyzeapi.ydble_utils import decrypt_ecb, encrypt_ecb

LOCK_UUID = "YD_BT1.b38ae5ebd9864864273db2c08fab2c28"


class FakeProxyClient:
    """Minimal BleakClient emulating bleak_esphome's notification accounting.

    ``start_notify`` raising on a double subscribe is the exact behaviour of
    ``bleak_esphome.backend.client.ESPHomeClient``, which rejects a handle already
    present in its ``_notify_cancels`` map. ``stop_notify`` silently no-ops when
    nothing is subscribed, matching that backend's documented contract.
    """

    def __init__(self):
        self.is_connected = True
        self.subscribed: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.writes: list[bytes] = []
        self.stop_notify_error: Exception | None = None
        self.con_type_error: Exception | None = None

    async def start_notify(self, uuid, callback):
        self.calls.append(("start", uuid))
        if uuid in self.subscribed:
            raise BleakError(
                f"Notifications are already enabled on characteristic:{uuid}"
            )
        self.subscribed.add(uuid)

    async def stop_notify(self, uuid):
        self.calls.append(("stop", uuid))
        if self.stop_notify_error is not None:
            raise self.stop_notify_error
        self.subscribed.discard(uuid)

    async def write_gatt_char(self, uuid, data, response=False):
        if uuid == YDBLE_CON_TYPE_UUID and self.con_type_error is not None:
            raise self.con_type_error
        self.calls.append(("write", uuid))
        self.writes.append((uuid, bytes(data)))

    async def disconnect(self):
        self.is_connected = False
        self.subscribed.clear()


def make_coordinator(client):
    """Build a real coordinator wired to a fake BLE client."""
    lock = MagicMock()
    lock.mac = LOCK_UUID
    lock.nickname = "Test Bolt"
    lock.ble_id = b"\x01\x02\x03\x04"
    lock.ble_token = b"\x05\x06\x07\x08"

    with patch("homeassistant.helpers.frame.report_usage"):
        from custom_components.wyzeapi.coordinator import WyzeLockBoltCoordinator

        coordinator = WyzeLockBoltCoordinator(MagicMock(), MagicMock(), lock)

    coordinator._mac = "28:68:47:E9:D0:7B"
    coordinator._bleak_client = client
    coordinator.async_update_listeners = MagicMock()
    return coordinator


async def finish_command(coordinator):
    """Emulate the lock reporting its new state, which ends the command."""
    coordinator._current_command = None


@pytest.mark.asyncio
async def test_back_to_back_commands_both_succeed():
    """The regression: a second command on a still-open connection must work.

    Before the fix this raised BleakError on the second unlock and the bolt never
    moved.
    """
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("unlock")
    await finish_command(coordinator)

    # Connection is deliberately still open, exactly as it is inside the
    # teardown window on a real lock.
    assert client.is_connected

    await coordinator.lock_unlock("lock")

    assert set(client.subscribed) == set(NOTIFY_UUIDS)
    # Two commands, so two challenge writes: the command actually ran.
    challenges = [u for u, _ in client.writes if u not in (YDBLE_CON_TYPE_UUID,)]
    assert len(challenges) == 2


@pytest.mark.asyncio
async def test_notifications_cleared_before_resubscribing():
    """stop_notify must precede start_notify for every notify characteristic."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("lock")

    for uuid in NOTIFY_UUIDS:
        stops = [
            i
            for i, (kind, u) in enumerate(client.calls)
            if kind == "stop" and u == uuid
        ]
        starts = [
            i
            for i, (kind, u) in enumerate(client.calls)
            if kind == "start" and u == uuid
        ]
        assert stops and starts, f"missing stop/start for {uuid}"
        assert min(stops) < min(starts), f"stop_notify must come first for {uuid}"


@pytest.mark.asyncio
async def test_clear_is_safe_on_a_fresh_connection():
    """A never-subscribed connection must not error, and must still subscribe."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("lock")

    assert set(client.subscribed) == set(NOTIFY_UUIDS)


@pytest.mark.asyncio
async def test_stop_notify_failure_does_not_block_the_command():
    """A refused stop_notify is logged and ignored, not propagated.

    The link may have died between the two calls, and start_notify surfaces any
    real problem.
    """
    client = FakeProxyClient()
    client.stop_notify_error = BleakError("not connected")
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("lock")

    assert set(client.subscribed) == set(NOTIFY_UUIDS)
    challenges = [u for u, _ in client.writes if u not in (YDBLE_CON_TYPE_UUID,)]
    assert len(challenges) == 1


@pytest.mark.asyncio
async def test_new_command_cancels_previous_teardown_timer():
    """The stale timer must not fire during the next command.

    Without cancellation the first command's timer clears _current_command midway
    through the second command, so a failure looks like a spurious state revert.
    """
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("unlock")
    first_timer = coordinator._disconnect_task
    await finish_command(coordinator)

    await coordinator.lock_unlock("lock")
    second_timer = coordinator._disconnect_task

    # cancel() only requests cancellation; let the loop deliver it.
    await asyncio.sleep(0)

    assert first_timer is not second_timer
    assert first_timer.cancelled() or first_timer.done()
    assert not second_timer.done()

    # The live command must still be marked in flight.
    assert coordinator._current_command == "lock"

    second_timer.cancel()


@pytest.mark.asyncio
async def test_teardown_timer_survives_long_enough_for_a_cold_connect():
    """A cold connect alone can eat 5.5s, so the budget must exceed the old 10s."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    delays = []
    original = coordinator._disconnect

    async def record(delay=0):
        delays.append(delay)
        return await original(delay=0)

    coordinator._disconnect = record
    await coordinator.lock_unlock("lock")
    await asyncio.sleep(0)

    assert delays and delays[0] > 10


@pytest.mark.asyncio
async def test_cancelling_teardown_leaves_the_connection_up():
    """Cancelling the timer must not disconnect the link the new command is using."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("unlock")
    await finish_command(coordinator)
    await coordinator.lock_unlock("lock")

    # Give the cancelled timer a chance to run its except branch.
    await asyncio.sleep(0)

    assert client.is_connected
    coordinator._disconnect_task.cancel()


@pytest.mark.asyncio
async def test_second_command_rejected_while_first_in_flight():
    """Pre-existing guard must still hold: no concurrent commands."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("unlock")
    # _current_command intentionally left set: the first command is mid-exchange.

    with pytest.raises(Exception, match="Waiting for unlock"):
        await coordinator.lock_unlock("lock")

    coordinator._disconnect_task.cancel()


@pytest.mark.asyncio
async def test_connection_type_declared_with_correct_payload():
    """The lock hangs up ~6s in without this, so it must be sent and be decryptable."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("lock")

    sent = [data for uuid, data in client.writes if uuid == YDBLE_CON_TYPE_UUID]
    assert len(sent) == 1, "connection type must be declared exactly once per command"
    assert len(sent[0]) == 16, "must be a single AES-128-ECB block"
    assert decrypt_ecb(LOCK_UUID[-16:].lower(), sent[0]) == YDBLE_CON_TYPE_PLAINTEXT


@pytest.mark.asyncio
async def test_connection_type_sent_after_rx_notify_before_challenge():
    """Mirrors the app: RX notify, then declare, then the exchange starts."""
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("lock")

    seq = [(k, u) for k, u in client.calls if k in ("start", "write")]
    rx = seq.index(("start", YDBLE_UART_RX_UUID))
    declare = seq.index(("write", YDBLE_CON_TYPE_UUID))
    challenge = next(
        i for i, (k, u) in enumerate(seq) if k == "write" and u != YDBLE_CON_TYPE_UUID
    )
    assert rx < declare < challenge


@pytest.mark.asyncio
async def test_command_proceeds_if_connection_type_write_fails():
    """A refused declaration must not abort the command; it may still finish in ~6s."""
    client = FakeProxyClient()
    client.con_type_error = BleakError("write failed")
    coordinator = make_coordinator(client)

    await coordinator.lock_unlock("lock")

    challenges = [u for u, _ in client.writes if u != YDBLE_CON_TYPE_UUID]
    assert len(challenges) == 1


@pytest.mark.asyncio
async def test_link_released_promptly_after_command_completes():
    """The lock allows one connection, so a finished command must free it fast.

    With the connection-type declaration the lock no longer hangs up on its own, so
    failing to release the link would lock the owner's phone out indefinitely.
    """
    client = FakeProxyClient()
    coordinator = make_coordinator(client)

    delays = []
    orig = coordinator._disconnect  # noqa: F841

    async def record(delay=0):
        # Record the scheduled delay without actually tearing the link down.
        delays.append(delay)

    coordinator._disconnect = record

    await coordinator.lock_unlock("lock")
    await asyncio.sleep(0)  # let the scheduled task start
    assert delays == [BOLT_COMMAND_TIMEOUT_SECONDS], "safety net armed on command start"

    # The lock reports "locked": byte 0 state, bytes 1:5 timestamp, 16-byte block.
    block = encrypt_ecb(LOCK_UUID[-16:].lower(), bytes([1, 0, 0, 0, 0]) + b"\x00" * 11)
    await coordinator._handle_state(None, bytearray(block))
    await asyncio.sleep(0)

    assert delays[-1] == BOLT_COMMAND_DISCONNECT_SECONDS
    assert delays[-1] < BOLT_COMMAND_TIMEOUT_SECONDS
    assert coordinator._current_command is None
