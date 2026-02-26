# MYNAH Wearable Firmware (ESP-IDF)

Firmware project path: `wearable/firmware/mynah_wearable`

## Scope of this draft
- Targets XIAO ESP32-C3
- Initializes I2C and reads MAX30102 FIFO samples
- Initializes I2S RX and captures microphone frames
- Debounced button input toggles voice-note session state (press = start, next press = stop)
- Records active voice-note audio to local SPIFFS as PCM (`/spiffs/note_<seq>.pcm`)
- BLE GATT sync service exposes object manifest + chunk fetch + commit/wipe + time sync
- Runtime logging is intentionally quiet (init/button/voice-note lifecycle + compact HR status)
- HR health emits one compact `hr_status` line every 5 seconds for live placement debugging
- HR path auto-recovers by re-running MAX30102 init after short consecutive FIFO read failures

This is a bring-up scaffold, not production firmware.

## Prerequisites
- ESP-IDF v5.x installed and exported
- USB connection to XIAO ESP32-C3

## Build
```bash
cd wearable/firmware/mynah_wearable
idf.py set-target esp32c3
idf.py build
```

## Flash and Monitor
```bash
idf.py -p <SERIAL_PORT> flash monitor
```

## Configurable pins and parameters
Edit `main/main.c`:
- I2C pins and bus speed
- I2S pins and sample rate
- Sensor addresses
- Button GPIO and debounce timing

## Current limitations / TODO
- HR algorithm is placeholder quality + synthetic BPM estimator
- Voice-note payload is raw PCM (no on-device compression yet)
- No battery reporting or secure key storage flow yet
