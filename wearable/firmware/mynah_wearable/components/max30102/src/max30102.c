#include "max30102.h"

#include "esp_log.h"

#define TAG "max30102"

#define MAX30102_REG_INTR_ENABLE_1   0x02
#define MAX30102_REG_INTR_ENABLE_2   0x03
#define MAX30102_REG_FIFO_WR_PTR     0x04
#define MAX30102_REG_OVF_COUNTER     0x05
#define MAX30102_REG_FIFO_RD_PTR     0x06
#define MAX30102_REG_FIFO_DATA       0x07
#define MAX30102_REG_FIFO_CONFIG     0x08
#define MAX30102_REG_MODE_CONFIG     0x09
#define MAX30102_REG_SPO2_CONFIG     0x0A
#define MAX30102_REG_LED1_PA         0x0C
#define MAX30102_REG_LED2_PA         0x0D
#define MAX30102_REG_PART_ID         0xFF

static esp_err_t reg_write(const max30102_config_t *cfg, uint8_t reg, uint8_t val)
{
    uint8_t payload[2] = {reg, val};
    return i2c_master_write_to_device(cfg->i2c_port, cfg->i2c_addr, payload, sizeof(payload), pdMS_TO_TICKS(100));
}

static esp_err_t reg_read(const max30102_config_t *cfg, uint8_t reg, uint8_t *buf, size_t len)
{
    return i2c_master_write_read_device(cfg->i2c_port, cfg->i2c_addr, &reg, 1, buf, len, pdMS_TO_TICKS(100));
}

esp_err_t max30102_init(const max30102_config_t *cfg)
{
    if (!cfg) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t part_id = 0;
    ESP_RETURN_ON_ERROR(reg_read(cfg, MAX30102_REG_PART_ID, &part_id, 1), TAG, "part id read failed");
    ESP_LOGI(TAG, "part id: 0x%02X", part_id);

    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_MODE_CONFIG, 0x40), TAG, "reset failed");
    vTaskDelay(pdMS_TO_TICKS(50));

    // FIFO pointers reset
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_FIFO_WR_PTR, 0x00), TAG, "fifo wr ptr set failed");
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_OVF_COUNTER, 0x00), TAG, "fifo ovf ptr set failed");
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_FIFO_RD_PTR, 0x00), TAG, "fifo rd ptr set failed");

    // FIFO config: sample avg 4, rollover enabled, almost full 17
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_FIFO_CONFIG, 0x4F), TAG, "fifo config failed");

    // SpO2 mode + HR mode (Red + IR)
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_MODE_CONFIG, 0x03), TAG, "mode config failed");

    // 100Hz sample rate, 411us pulse width (18-bit)
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_SPO2_CONFIG, 0x27), TAG, "spo2 config failed");

    // LED current defaults (tune later on hardware)
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_LED1_PA, 0x24), TAG, "red led config failed");
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_LED2_PA, 0x24), TAG, "ir led config failed");

    // Disable interrupts for polling mode in this scaffold
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_INTR_ENABLE_1, 0x00), TAG, "int1 disable failed");
    ESP_RETURN_ON_ERROR(reg_write(cfg, MAX30102_REG_INTR_ENABLE_2, 0x00), TAG, "int2 disable failed");

    return ESP_OK;
}

esp_err_t max30102_read_sample(const max30102_config_t *cfg, max30102_sample_t *out)
{
    if (!cfg || !out) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t raw[6] = {0};
    ESP_RETURN_ON_ERROR(reg_read(cfg, MAX30102_REG_FIFO_DATA, raw, sizeof(raw)), TAG, "fifo read failed");

    out->red = ((uint32_t)(raw[0] & 0x03) << 16) | ((uint32_t)raw[1] << 8) | raw[2];
    out->ir = ((uint32_t)(raw[3] & 0x03) << 16) | ((uint32_t)raw[4] << 8) | raw[5];
    out->valid = (out->red > 1000U && out->ir > 1000U);

    return ESP_OK;
}
