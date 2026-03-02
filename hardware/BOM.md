# TeensyBat USB - Bill of Materials

Teensy 4.1 as a USB ultrasonic audio device, streaming 192 kHz audio to a laptop for real-time analysis.

Budget: ~60-80 EUR

## Core Electronics

| # | Part | Qty | Price (EUR) | Source | Notes |
|---|---|---|---|---|---|
| 1 | Teensy 4.1 | 1 | ~35 | Lextronic.fr, Exp-Tech.de, PJRC.com | 600 MHz Cortex-M7, USB |
| 2 | PJRC Audio Shield Rev D | 1 | ~15 | Lextronic.fr, Exp-Tech.de | SGTL5000 codec, I2S ADC |
| 3 | Knowles SPU0410LR5H on PCB | 1 | ~10 | Teensybat.com, Micbooster.com | Analog MEMS mic, flat to ~80 kHz. Pre-soldered breakout recommended |

## Passive Components

| # | Part | Qty | Price (EUR) | Source | Notes |
|---|---|---|---|---|---|
| 4 | Resistor 2.2 kΩ | 1 | - | Any | Mic bias resistor |
| 5 | Capacitor 100 nF ceramic | 2 | - | Any | Decoupling |
| 6 | Capacitor 1 µF ceramic | 1 | - | Any | Mic output coupling |
| 7 | Pin headers (male) | 1 strip | ~1 | Any | For Teensy + Audio Shield stacking |

## Prototyping

| # | Part | Qty | Price (EUR) | Source | Notes |
|---|---|---|---|---|---|
| 8 | Breadboard (830 points) | 1 | ~4 | Lextronic.fr, Amazon.fr | For prototyping before final assembly |
| 9 | Jumper wires | 1 set | ~3 | Any | Male-male and male-female |
| 10 | USB micro cable | 1 | ~3 | Any | Data + power from laptop |

## Total: ~60-80 EUR

## Not Needed (laptop handles these)

- ~~OLED display~~ — spectrogram displays on laptop
- ~~SD card~~ — recordings saved to laptop
- ~~Battery / charger~~ — powered via USB from laptop
- ~~Enclosure~~ — optional, breadboard prototype is fine for development

## Optional (later)

| Part | Price (EUR) | Notes |
|---|---|---|
| Small enclosure / project box | ~5-10 | For field use |
| Perfboard | ~2 | Permanent assembly after prototyping |
| Short USB cable (30 cm) | ~3 | Cleaner field setup |
| Headphones (3.5mm) | - | Listen via Audio Shield headphone jack |

## Alternative Microphones

If the SPU0410LR5H is unavailable:

| Mic | Type | Range | Pros | Cons |
|---|---|---|---|---|
| **Knowles SPH0641LU4H-1** | Digital PDM | to ~80 kHz | Still in production | Needs PDM decoding, more complex |
| **ICS-40730** (TDK InvenSense) | Analog MEMS | to ~70 kHz | Easy analog interface | Slightly lower bandwidth |
| **Electret + preamp** | Analog | to ~100 kHz+ | Widest bandwidth | Bulkier, needs custom preamp circuit |

## Where to Buy (France / Europe)

- **Teensy + Audio Shield**: [Lextronic.fr](https://www.lextronic.fr) (France, fast), [Exp-Tech.de](https://www.exp-tech.de) (Germany), [PJRC.com](https://www.pjrc.com) (US)
- **MEMS mic breakout**: [Teensybat.com](https://www.teensybat.com), [Micbooster.com](https://micbooster.com)
- **General components**: Amazon.fr, Reichelt.de, AliExpress
