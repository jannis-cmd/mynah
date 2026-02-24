#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c.h"
#include "esp_err.h"

typedef struct {
    uint32_t red;
    uint32_t ir;
    bool valid;
} max30102_sample_t;

typedef struct {
    i2c_port_t i2c_port;
    uint8_t i2c_addr;
} max30102_config_t;

esp_err_t max30102_init(const max30102_config_t *cfg);
esp_err_t max30102_read_sample(const max30102_config_t *cfg, max30102_sample_t *out);
