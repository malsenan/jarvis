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
DEFAULT_DEVICE = 3  # the "default" entry above, what index None resolves to


@pytest.fixture(autouse=True)
def fake_query_devices(monkeypatch):
    """Stand in for sounddevice's device table, in both of its call shapes.

    query_devices() returns the whole table (find_device); query_devices(index,
    kind) returns one device, with None meaning the system default
    (describe_device).
    """
    def query(index="all", kind=None):
        if index == "all":
            return FAKE_DEVICES
        return FAKE_DEVICES[DEFAULT_DEVICE] if index is None else FAKE_DEVICES[index]

    monkeypatch.setattr(audio_devices.sd, "query_devices", query)


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


# --- Naming the device we picked -----------------------------------------
#
# describe_device is what makes the startup print useful: it turns an index
# (or None) into a name, and resolves "default" through PipeWire so the line
# says which microphone that actually is.

ATR = "alsa_input.usb-Conference_USB_microphone_ATR4697-USB-00.mono-fallback"


@pytest.fixture
def fake_pipewire(monkeypatch):
    monkeypatch.setattr(audio_devices, "pipewire_default", lambda kind: ATR)


def test_named_device_is_reported_as_is(fake_pipewire):
    # A real device name already says which hardware it is; leave it alone.
    assert audio_devices.describe_device(2, "input") == "ATR4697-USB: USB Audio"


def test_default_device_is_resolved_through_pipewire(fake_pipewire):
    # "default" alone tells the user nothing, so name what is behind it.
    assert audio_devices.describe_device(None, "input") == f"default -> {ATR}"


def test_pipewire_default_asks_pactl_for_the_right_thing(monkeypatch):
    asked = []

    def fake_run(command, **kwargs):
        asked.append(command)
        return type("Result", (), {"stdout": ATR + "\n"})()

    monkeypatch.setattr(audio_devices.subprocess, "run", fake_run)
    assert audio_devices.pipewire_default("input") == ATR
    audio_devices.pipewire_default("output")
    assert asked == [
        ["pactl", "get-default-source"],
        ["pactl", "get-default-sink"],
    ]
