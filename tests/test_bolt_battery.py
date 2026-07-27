"""Tests for the Wyze Lock Bolt battery block decoder.

The Bolt's 0x2A19 characteristic is not a standard one-byte BLE Battery Level. It
returns a 16-byte AES-128-ECB block encrypted with the same key used for the lock
state characteristic. Layout of the decrypted block, per the Wyze app's own decode
(``C21643f.java`` battery callback plus ``C21552t.m67888k``):

    byte 0      battery percentage, valid range 1..100
    bytes 11:16 ASCII marker "loock"

These tests build blocks with the real cipher and assert the decoder recovers the
expected values, so they fail if the layout or validation is ever broken.
"""

from custom_components.wyzeapi.ydble_utils import encrypt_ecb, parse_battery_block

# 16 chars, matching the lock UUID's trailing 16 used as the AES key.
KEY = "273db2c08fab2c28"


def _block(percent: int, marker: bytes = b"loock", pad: bytes = b"\x00" * 6) -> bytes:
    """Build a plaintext status block, then encrypt it the way the lock would."""
    plaintext = bytes([percent]) + b"\x00" * 4 + pad[:6] + marker
    assert len(plaintext) == 16, f"block must be one AES block, got {len(plaintext)}"
    return encrypt_ecb(KEY, plaintext)


def test_decodes_reported_percentage():
    """98% is what the Wyze app showed for this lock, so use it as the worked example."""
    assert parse_battery_block(KEY, _block(98)) == 98


def test_decodes_boundary_values():
    assert parse_battery_block(KEY, _block(1)) == 1
    assert parse_battery_block(KEY, _block(100)) == 100


def test_rejects_zero_and_out_of_range():
    """The app treats 0 and >100 as invalid rather than reporting a bogus level."""
    assert parse_battery_block(KEY, _block(0)) is None
    assert parse_battery_block(KEY, _block(101)) is None
    assert parse_battery_block(KEY, _block(255)) is None


def test_rejects_wrong_marker():
    """A wrong key decrypts to noise; the marker is what catches it."""
    assert parse_battery_block(KEY, _block(50, marker=b"xxxxx")) is None


def test_rejects_wrong_key():
    """Decrypting with a different key must not yield a plausible percentage."""
    other = "ffffffffffffffff"
    assert parse_battery_block(other, _block(98)) is None


def test_rejects_short_block():
    """A standard one-byte Battery Level read must not be misread as a status block."""
    assert parse_battery_block(KEY, b"\x62") is None


def test_ignores_padding_contents():
    """Only byte 0 and the marker are load-bearing; padding must not affect decoding."""
    assert parse_battery_block(KEY, _block(77, pad=b"\x01\x02\x03\x04\x05\x06")) == 77
