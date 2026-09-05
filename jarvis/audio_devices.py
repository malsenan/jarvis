"""Find audio devices by name instead of by index, and say which we picked.

Device indices are not stable: they shift when a USB device is plugged in,
when PipeWire restarts, or simply between boots. So callers give a
case-insensitive substring of the device *name* and we look up whatever index
that name has right now, at startup.

Run this module directly to see every device on the system:

    .venv/bin/python -m jarvis.audio_devices
"""

import subprocess

import sounddevice as sd


def find_device(name_fragment: str | None, kind: str) -> int | None:
    """Return the sounddevice index of the device whose name contains
    `name_fragment`, or None to use the system default.

    Args:
        name_fragment: case-insensitive substring of the device name
                       (e.g. "ATR4697"), or None for the system default.
        kind: "input" or "output" — only devices with channels in that
              direction are considered.

    Raises:
        LookupError: no device matches; the message lists what is available.
    """
    if name_fragment is None:
        return None

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()

    matches = [
        (index, device)
        for index, device in enumerate(devices)
        if name_fragment.lower() in device["name"].lower() and device[channel_key] > 0
    ]

    if not matches:
        available = ", ".join(
            f"'{d['name']}'" for d in devices if d[channel_key] > 0
        )
        raise LookupError(
            f"No {kind} device matching '{name_fragment}'. "
            f"Available {kind} devices: {available}"
        )

    # A physical card often shows up more than once (raw ALSA entry plus a
    # "plug" entry). The first match works in practice; we mention the rest
    # so a surprising choice is easy to diagnose.
    if len(matches) > 1:
        names = ", ".join(f"[{i}] {d['name']}" for i, d in matches)
        print(f"Note: several {kind} devices match '{name_fragment}': {names}. "
              f"Using the first one.")

    return matches[0][0]


def describe_device(index: int | None, kind: str) -> str:
    """Name the device we are about to open, for printing at startup.

    `index` is what find_device returned, so None means the system default.
    PortAudio calls that device "default" no matter what is behind it, so
    when we see that name we ask PipeWire which microphone or speaker it
    currently points at — otherwise "default" tells you nothing.

    Args:
        index: sounddevice device index, or None for the system default.
        kind: "input" or "output".
    """
    name = sd.query_devices(index, kind)["name"]
    if name in ("default", "pipewire"):
        return f"{name} -> {pipewire_default(kind)}"
    return name


def pipewire_default(kind: str) -> str:
    """Ask PipeWire which microphone/speaker it is currently defaulting to."""
    command = "get-default-source" if kind == "input" else "get-default-sink"
    result = subprocess.run(
        ["pactl", command], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


if __name__ == "__main__":
    # Prints one line per device with its index, name and channel counts.
    print(sd.query_devices())
