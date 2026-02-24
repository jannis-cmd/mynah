#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "driver/i2c.h"
#include "driver/i2s.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "max30102.h"
#include "nvs_flash.h"

#define TAG "mynah_fw"

// Seeed XIAO ESP32-C3 default I2C pins: D4(GPIO6) SDA, D5(GPIO7) SCL.
#define I2C_PORT I2C_NUM_0
#define I2C_SDA_PIN GPIO_NUM_6
#define I2C_SCL_PIN GPIO_NUM_7
#define I2C_FREQ_HZ 400000

#define MAX30102_I2C_ADDR 0x57

// DFRobot I2S MEMS microphone (SEN0327) pin mapping (configurable).
#define I2S_PORT I2S_NUM_0
#define I2S_BCLK_PIN GPIO_NUM_10
#define I2S_WS_PIN GPIO_NUM_21
#define I2S_SD_PIN GPIO_NUM_20
#define I2S_SAMPLE_RATE 16000
#define I2S_BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_16BIT
#define I2S_DMA_BUF_COUNT 4
#define I2S_DMA_BUF_LEN 256

static max30102_config_t g_hr_cfg = {
    .i2c_port = I2C_PORT,
    .i2c_addr = MAX30102_I2C_ADDR,
};

static esp_err_t i2c_init(void)
{
    const i2c_config_t cfg = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_SDA_PIN,
        .scl_io_num = I2C_SCL_PIN,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
        .clk_flags = 0,
    };

    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_PORT, &cfg), TAG, "i2c_param_config failed");
    return i2c_driver_install(I2C_PORT, cfg.mode, 0, 0, 0);
}

static esp_err_t i2s_mic_init(void)
{
    const i2s_config_t i2s_cfg = {
        .mode = I2S_MODE_MASTER | I2S_MODE_RX,
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = I2S_DMA_BUF_COUNT,
        .dma_buf_len = I2S_DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
        .mclk_multiple = I2S_MCLK_MULTIPLE_256,
        .bits_per_chan = I2S_BITS_PER_CHAN_DEFAULT,
    };

    const i2s_pin_config_t pin_cfg = {
        .mck_io_num = I2S_PIN_NO_CHANGE,
        .bck_io_num = I2S_BCLK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN,
    };

    ESP_RETURN_ON_ERROR(i2s_driver_install(I2S_PORT, &i2s_cfg, 0, NULL), TAG, "i2s_driver_install failed");
    return i2s_set_pin(I2S_PORT, &pin_cfg);
}

static int estimate_bpm(const max30102_sample_t *s)
{
    // Placeholder for a proper pulse-processing pipeline.
    // Returns a rough synthetic bpm in valid range if signal quality looks acceptable.
    if (!s->valid) {
        return -1;
    }

    uint32_t composite = (s->red + s->ir) / 2U;
    int bpm = 55 + (int)(composite % 45U);
    return bpm;
}

static void hr_task(void *arg)
{
    (void)arg;
    while (1) {
        max30102_sample_t sample = {0};
        esp_err_t err = max30102_read_sample(&g_hr_cfg, &sample);
        if (err == ESP_OK) {
            int bpm = estimate_bpm(&sample);
            int quality = sample.valid ? 1 : 0;
            ESP_LOGI(TAG, "hr_sample ts=%lld red=%lu ir=%lu bpm=%d quality=%d",
                     esp_timer_get_time() / 1000,
                     (unsigned long)sample.red,
                     (unsigned long)sample.ir,
                     bpm,
                     quality);
        } else {
            ESP_LOGW(TAG, "max30102_read_sample failed: %s", esp_err_to_name(err));
        }

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

static void mic_task(void *arg)
{
    (void)arg;
    int16_t buf[512];

    while (1) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_read(I2S_PORT, buf, sizeof(buf), &bytes_read, pdMS_TO_TICKS(1000));
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "i2s_read failed: %s", esp_err_to_name(err));
            continue;
        }

        size_t n = bytes_read / sizeof(int16_t);
        if (n == 0) {
            continue;
        }

        double sum_sq = 0.0;
        for (size_t i = 0; i < n; i++) {
            double x = (double)buf[i];
            sum_sq += x * x;
        }
        double rms = sqrt(sum_sq / (double)n);

        ESP_LOGI(TAG, "mic_frame ts=%lld samples=%u rms=%.2f",
                 esp_timer_get_time() / 1000,
                 (unsigned)n,
                 rms);
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());

    ESP_ERROR_CHECK(i2c_init());
    ESP_ERROR_CHECK(max30102_init(&g_hr_cfg));

    ESP_ERROR_CHECK(i2s_mic_init());

    ESP_LOGI(TAG, "mynah wearable scaffold started");

    xTaskCreate(hr_task, "hr_task", 4096, NULL, 5, NULL);
    xTaskCreate(mic_task, "mic_task", 4096, NULL, 4, NULL);
}
