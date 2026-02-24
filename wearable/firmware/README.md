# MINAH Wearable Firmware (ESP-IDF)

Firmware project path: `wearable/firmware/mynah_wearable`

## Scope of this draft
- Targets XIAO ESP32-C3
- Initializes I2C and reads MAX30102 FIFO samples
- Initializes I2S RX and captures microphone frames
- Logs HR sample payload and microphone RMS for bring-up/debug

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

## Current limitations / TODO
- BLE GATT service and sync flow not implemented yet
- HR algorithm is placeholder quality + synthetic BPM estimator
- No voice-note file storage/encoding pipeline yet
- No button controls, battery reporting, or secure key storage flow yet
