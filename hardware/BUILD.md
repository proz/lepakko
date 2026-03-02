# TeensyBat USB - Build Notes

## Architecture

The Teensy acts as a USB audio device streaming 192 kHz audio to a laptop.
All processing (spectrogram, detection, classification) runs in Python on the laptop.

```
┌─────────────┐
│  MEMS Mic   │
│ SPU0410LR5H │
└──────┬──────┘
       │ analog audio
┌──────▼──────┐
│ Audio Shield │
│  SGTL5000   │◄──── optional: headphone out (3.5mm)
│  (I2S ADC)  │
└──────┬──────┘
       │ I2S digital audio
┌──────▼──────┐        USB 192 kHz audio       ┌──────────────┐
│  Teensy 4.1 │ ──────────────────────────────► │    Laptop    │
│  Cortex-M7  │    (appears as sound card)      │              │
│  600 MHz    │                                 │  Python      │
└─────────────┘                                 │  ├ spectrogram│
                                                │  ├ detection  │
                                                │  └ classify   │
                                                └──────────────┘
```

## Wiring

### Audio Shield → Teensy 4.1

The Audio Shield stacks directly onto Teensy 4.1 via pin headers. No wiring needed — just solder headers and stack them.

Audio Shield uses these Teensy pins (directly through stacking):
- I2S: 7 (RX), 8 (TX), 20 (LRCLK), 21 (BCLK), 23 (MCLK)
- I2C control: 18 (SDA), 19 (SCL)

### MEMS Mic → Audio Shield

| Mic Pin | Connect To | Notes |
|---|---|---|
| VDD | 3.3V | From Audio Shield or Teensy |
| GND | GND | |
| OUT | Audio Shield LINE IN (left) | Through 1 µF coupling capacitor |
| - | 2.2 kΩ from VDD to OUT | Bias resistor |

Add a 100 nF decoupling cap between VDD and GND, close to the mic.

If using the Teensybat.com breakout board, the bias resistor and caps are already on the PCB — just connect VDD, GND, and OUT.

### Power

Powered entirely via USB from the laptop. No battery needed.

## Firmware

The Teensy needs firmware that:
1. Reads audio from the Audio Shield (I2S, 192 kHz)
2. Streams it over USB as a USB Audio device

### Option A: laiudm 192k USB Audio patch (recommended)

Reference: [laiudm/Teensy4-192k-USB-Audio](https://github.com/laiudm/laiudm-Teensy4-192k-USB-Audio)

This patches Teensyduino to support 192 kHz USB audio. The Teensy appears as a standard 192 kHz sound card — no custom drivers needed on the laptop.

Setup:
1. Install Arduino IDE + Teensyduino
2. Apply the 192k USB audio patch from the repo
3. Set board to Teensy 4.1, USB Type: Audio
4. Set CPU speed to 600 MHz
5. Write a minimal sketch that routes I2S input to USB output
6. Upload

### Option B: Custom USB Serial streaming

If USB Audio class doesn't work reliably at 192 kHz, fall back to streaming raw samples over USB Serial. The Python side reads from the serial port and buffers the audio. More work but guaranteed throughput.

### Sample Rate

The SGTL5000 officially supports up to 96 kHz. The bat detector community overclocks it to 192 kHz by modifying I2S clock dividers. This is well-tested.

At 192 kHz sample rate:
- Nyquist frequency = 96 kHz
- Covers most European species
- Does NOT cover lesser horseshoe bat (~110 kHz) — would need 250+ kHz

### Reference projects

- [laiudm/Teensy4-192k-USB-Audio](https://github.com/laiudm/laiudm-Teensy4-192k-USB-Audio) — 192 kHz USB audio patch
- [alex6679/teensy-4-usbAudio](https://github.com/alex6679/teensy-4-usbAudio) — Multi-channel USB audio for Teensy 4.x
- [CorBer/teensy_batdetector](https://github.com/CorBer/teensy_batdetector) — Full standalone bat detector (reference for audio setup)
- [DD4WH/Teensy-Bat-Detector](https://github.com/DD4WH/Teensy-Bat-Detector) — Another standalone bat detector

## Assembly Steps

### Prototyping (breadboard)

1. Solder pin headers to Teensy 4.1 (bottom side)
2. Solder pin headers to Audio Shield (top and bottom)
3. Stack Audio Shield onto Teensy
4. Plug the stack into breadboard
5. Wire MEMS mic breakout to Audio Shield LINE IN (3 wires: VDD, GND, OUT)
6. Connect USB to laptop
7. Flash firmware
8. Verify: laptop should see a new USB audio input device at 192 kHz

### Testing without mic

You can test the USB audio streaming first using the Audio Shield's LINE IN jack with a regular audio source (phone, signal generator). This verifies the firmware and USB link before adding the MEMS mic.

### Field assembly (later)

Once prototyping works:
1. Solder permanent connections on perfboard
2. Mount in small enclosure
3. Route mic through hole in top of enclosure
4. Short USB cable to laptop

## Troubleshooting

- **No audio device on laptop**: Check USB Type is set to "Audio" in Arduino IDE. Try a different USB cable (some are charge-only).
- **Audio but no ultrasonic signal**: Verify mic is connected to LINE IN, not MIC IN. Check bias resistor.
- **Noise / interference**: Keep mic wires short (<5 cm). Add ferrite bead on USB cable if needed.
- **Dropouts at 192 kHz**: Try a USB 2.0 port directly (not through a hub). The 192k patch may need USB buffer size tuning.
