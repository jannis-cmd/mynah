# MYNAH Wearable

This directory contains wearable hardware and firmware assets for the MYNAH offline personal intelligence project.

## Hardware Target
- MCU board: Seeed Studio XIAO ESP32-C3
- HR/SpO2 sensor (I2C): DFRobot Heart Rate and Oximeter Sensor V2 (SEN0344, MAX30102)
- Microphone (I2S): DFRobot I2S MEMS microphone module (SEN0327)

## Current Firmware Status
- ESP-IDF scaffold created at `wearable/firmware/mynah_wearable`
- I2C init + MAX30102 basic polling implemented
- I2S microphone capture task scaffold implemented
- BLE sync/profile is not implemented yet in this draft

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

## Build and Flash
See `wearable/firmware/README.md` for ESP-IDF setup and flash commands.

## References
- XIAO ESP32-C3 Getting Started: https://wiki.seeedstudio.com/XIAO_ESP32C3_Getting_Started/
- DFRobot SEN0344 (MAX30102): https://wiki.dfrobot.com/Heart_Rate_and_Oximeter_Sensor_V2_SKU_SEN0344
- DFRobot I2S MEMS mic: https://www.dfrobot.com/product-1954.html
