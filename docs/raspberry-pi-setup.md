# Raspberry Pi Appliance Setup

This guide walks through configuring Meeko as a dedicated, screenless voice assistant appliance on a Raspberry Pi 5 with the ReSpeaker XVF3800 USB microphone array.

Before starting hardware configuration, make sure you have cloned the repository, installed dependencies, and configured your `.env` file as described in [Getting Started](../README.md#getting-started).

---

## Hardware Requirements

- **Raspberry Pi 5** (8 GB recommended) with active cooling (Raspberry Pi Active Cooler or equivalent).
- **NVMe SSD** via PCIe HAT (recommended) — micro-SD cards are significantly slower for ONNX model loading and SQLite disk I/O.
- **[ReSpeaker XVF3800](https://www.seeedstudio.com/ReSpeaker-USB-Mic-Array-p-4247.html) USB 4-Mic Array** with hardware acoustic echo cancellation (AEC), beamforming, and an onboard WS2812 LED ring.
- **Powered external speaker** connected via a standard 3.5mm audio cable.
- **System dependencies**: PortAudio (`sudo apt install portaudio19-dev`).

> [!IMPORTANT]
> The speaker **must be plugged into the ReSpeaker XVF3800's 3.5mm audio jack**, not the Raspberry Pi's audio output. The XMOS XVF3800 chip relies on the speaker audio signal as its far-end reference to cancel out echo and enable barge-in while Meeko is speaking.

---

## 1. Tune AEC Sensitivity (One-Time Setup)

On the Raspberry Pi + XVF3800 setup, the chip's default factory AEC tuning suppresses near-end speech aggressively during far-end playback, preventing barge-in (your voice will not reach Deepgram while Meeko is talking). You must raise the double-talk sensitivity once and persist it to flash memory.

1. Download the `xvf_host` binary from the [reSpeaker XVF3800 repository](https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/tree/master/host_control) (`host_control/rpi_64bit/`).
2. Make it executable and write the tuning parameters to flash:

   ```bash
   chmod +x ./xvf_host
   sudo ./xvf_host PP_DTSENSITIVE 12
   sudo ./xvf_host SAVE_CONFIGURATION 1
   ```

`PP_DTSENSITIVE 12` enables the chip's extra near-end speech detector and biases the AEC toward double-talk performance. `SAVE_CONFIGURATION 1` writes the value to flash so it survives power cycles.

*(To revert to factory defaults at any time: `sudo ./xvf_host CLEAR_CONFIGURATION 1` and reboot).*

---

## 2. USB Permissions for LED Control

Meeko communicates with the XVF3800's WS2812 LED ring directly over libusb. Linux defaults USB vendor control endpoints to root-only access. Install the shipped udev rule once so a non-root user (in the `plugdev` group) can control the LEDs:

```bash
sudo cp scripts/99-meeko-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo udevadm trigger --action=add --subsystem-match=usb
```

No reboot or replugging is required.

---

## 3. What the LED Ring Shows

On a headless appliance with no display, the XVF3800 LED ring serves as Meeko's primary visual interface, mirroring the internal orchestrator state:

| Ring Appearance | Internal State | Meaning |
|---|---|---|
| **Off** | `IDLE` | Waiting for the wake word ("Hey Meeko"). Audio is evaluated locally; nothing is sent off-device. |
| **Solid cyan** | `LISTENING` | Awake and ready for your turn. |
| **Solid brighter cyan** | `LISTENING_ACTIVE` | Actively hearing your speech right now. |
| **Breathing blue** | `PROCESSING` | Claude is generating a response (or searching the web). Speak to cancel the turn. |
| **Solid green** | `SPEAKING` | Speaking back via TTS. Speak over it to interrupt (barge-in). |
| **Breathing red (~3s)** | Error | A turn failed; Meeko returns to `LISTENING` immediately afterward. |

LED colors and pulse speeds can be customized in `PALETTE` at the top of `meeko/leds.py`. To disable LED control entirely, set `led_disabled = true` under `[system]` in `meeko.toml`.

---

## 4. Run Meeko at Boot (systemd User Service)

To have Meeko start automatically when the Raspberry Pi boots—and remain running in the background without requiring an SSH session—install the provided systemd user service.

1. Create your user systemd directory and copy the unit file:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp scripts/meeko.service ~/.config/systemd/user/
   ```

   *(Note: The unit file assumes the repository is located at `~/meeko`. If you cloned it elsewhere, edit `WorkingDirectory=` in the service file).*

2. Enable lingering so systemd starts your user manager at boot time without a GUI or SSH login:

   ```bash
   sudo loginctl enable-linger $USER
   ```

3. Enable and start the service:

   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now meeko
   ```

### Managing the Service

- **Tail live logs**: `journalctl --user-unit=meeko -f`
- **View logs since boot**: `journalctl --user-unit=meeko -b`
- **Check service status**: `systemctl --user status meeko`
- **Restart service** (e.g. after config or code updates): `systemctl --user restart meeko`
- **Stop service**: `systemctl --user stop meeko`

---

## 5. Offline Deployments & Wake-Word Cache

On its first run, openWakeWord downloads preprocessor models (~6.7 MB) into its local cache. If your Raspberry Pi will operate on a network without internet access (or with restricted outbound traffic), run the preprocessor downloader on a connected machine first:

```bash
uv run python -m meeko.wake_word
```

See [wake-word.md](wake-word.md) for further details on wake-word training and model caching.
