# MYNAH Wearable

This directory contains wearable hardware and firmware assets for the MYNAH offline personal intelligence project.

## Hardware Target
- MCU board: Seeed Studio XIAO ESP32-C3
- HR/SpO2 sensor (I2C): DFRobot Heart Rate and Oximeter Sensor V2 (SEN0344, MAX30102)
- Microphone (I2S): DFRobot I2S MEMS microphone module (SEN0327)

## Current Firmware Status
- ESP-IDF scaffold created at `wearable/firmware/mynah_wearable`
- I2C init + MAX30102 polling and on-device HR buffering implemented
- I2S microphone capture + button-toggled voice-note recording to SPIFFS implemented
- Debounced button toggle implemented (`press -> voice_note_started`, next press -> `voice_note_stopped`)
- Runtime logs are reduced to lifecycle/status lines to keep serial monitor readable.
- HR emits compact `hr_status` every 5 seconds (reads/valid/errors/recoveries).
- BLE GATT sync is implemented:
  - `device_info`, `status`, `manifest`, `fetch_req`, `fetch_data`, `commit_sync`, `wipe_confirm`, `time_sync`
  - compute can dump unsynced HR + voice-note objects, verify SHA, then commit/wipe on-device buffers

## Suggested Wiring (current draft)

### XIAO ESP32-C3 -> MAX30102 (SEN0344)
- `3V3` -> `VCC`
- `GND` -> `GND`
- `GPIO6 (D4)` -> `SDA`
- `GPIO7 (D5)` -> `SCL`

### XIAO ESP32-C3 -> I2S Mic (SEN0327)
- `3V3` -> `VCC`
- `GND` -> `GND`
- `GPIO10 (D10)` -> `SCK/BCLK`
- `GPIO21 (D6)` -> `WS/LRCLK`
- `GPIO20 (D7)` -> `SD`

Note: microphone pin mapping is configurable in firmware and may need adjustment based on your exact wiring.

### XIAO ESP32-C3 -> Button (voice note toggle)
- `GPIO9 (D9)` -> one button pin
- `GND` -> other button pin
- Input mode uses internal pull-up (`active-low`); avoid holding during reset/boot

## Serial Runtime Output
After flashing/reset, firmware prints:
- BLE startup/advertising state
- button events and voice-note start/stop metadata
- periodic `hr_status` lines (`reads`, `valid`, `errs`, `recoveries`, latest values)

## Build and Flash
See `wearable/firmware/README.md` for ESP-IDF setup and flash commands.

## References
- XIAO ESP32-C3 Getting Started: https://wiki.seeedstudio.com/XIAO_ESP32C3_Getting_Started/
- DFRobot SEN0344 (MAX30102): https://wiki.dfrobot.com/Heart_Rate_and_Oximeter_Sensor_V2_SKU_SEN0344
- DFRobot I2S MEMS mic: https://www.dfrobot.com/product-1954.html
