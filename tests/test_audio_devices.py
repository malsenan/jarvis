"""Tests for name-based audio device lookup.

sounddevice.query_devices is monkeypatched with a fake device table, so
these run without touching any real audio hardware.
"""

import pytest

from jarvis import audio_devices

FAKE_DEVICES = [
    {"name": "HDA ATI HDMI: HDMI 0", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "HD-Audio Generic: ALC897 Analog", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "ATR4697-USB: USB Audio", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "default", "max_input_channels": 32, "max_output_channels": 32},
]


@pytest.fixture(autouse=True)
def fake_query_devices(monkeypatch):
    monkeypatch.setattr(audio_devices.sd, "query_devices", lambda: FAKE_DEVICES)


def test_none_means_system_default():
    assert audio_devices.find_device(None, "input") is None
    assert audio_devices.find_device(None, "output") is None


def test_finds_mic_by_case_insensitive_substring():
    assert audio_devices.find_device("atr4697", "input") == 2


def test_finds_output_by_substring():
    assert audio_devices.find_device("ALC897", "output") == 1


def test_output_only_device_is_not_offered_as_input():
    # The HDMI device has zero input channels, so asking for it as an
    # input must fail rather than return an unusable device.
    with pytest.raises(LookupError):
        audio_devices.find_device("HDMI", "input")


def test_unknown_name_raises_with_available_devices_listed():
    with pytest.raises(LookupError) as error:
        audio_devices.find_device("does-not-exist", "input")
    # The error message should help the user pick a real device.
    assert "ATR4697" in str(error.value)
