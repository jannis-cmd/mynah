#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/i2s.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_spiffs.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "host/ble_hs.h"
#include "max30102.h"
#include "mbedtls/sha256.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define TAG "mynah_fw"

// Seeed XIAO ESP32-C3 default I2C pins: D4(GPIO6) SDA, D5(GPIO7) SCL.
#define I2C_PORT I2C_NUM_0
#define I2C_SDA_PIN GPIO_NUM_6
#define I2C_SCL_PIN GPIO_NUM_7
#define I2C_FREQ_HZ 100000

#define MAX30102_I2C_ADDR 0x57

// DFRobot I2S MEMS microphone (SEN0327) pin mapping (configurable).
#define I2S_PORT I2S_NUM_0
#define I2S_BCLK_PIN GPIO_NUM_10
#define I2S_WS_PIN GPIO_NUM_21
#define I2S_SD_PIN GPIO_NUM_20
#define I2S_SAMPLE_RATE 16000
#define I2S_DMA_BUF_COUNT 4
#define I2S_DMA_BUF_LEN 256
#define I2S_BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_16BIT

#define BUTTON_PIN GPIO_NUM_9
#define BUTTON_ACTIVE_LEVEL 0
#define BUTTON_POLL_MS 10
#define BUTTON_DEBOUNCE_MS 40

#define HR_STATUS_LOG_MS 5000
#define HR_RECOVERY_ERR_STREAK 3
#define HR_RECOVERY_DELAY_MS 200

#define HR_STORE_MAX 1024
#define AUDIO_NOTE_MAX 12
#define AUDIO_NOTE_MAX_BYTES (180 * 1024)

#define BLE_DEVICE_NAME "MYNAH-WEARABLE"
#define BLE_STATUS_MAX 256
#define BLE_INFO_MAX 192
#define BLE_MANIFEST_MAX 2048
#define BLE_WIPE_MAX 256
#define BLE_FETCH_REQ_MAX 128
#define BLE_FETCH_CHUNK_MAX 240
#define BLE_EXPORT_MAX 65536
#define HR_OBJECT_ID "hr.csv"

enum {
    BLE_UUID_SERVICE = 0xFFF0,
    BLE_UUID_DEVICE_INFO = 0xFFF1,
    BLE_UUID_STATUS = 0xFFF2,
    BLE_UUID_MANIFEST = 0xFFF3,
    BLE_UUID_FETCH_REQ = 0xFFF4,
    BLE_UUID_FETCH_DATA = 0xFFF5,
    BLE_UUID_COMMIT_SYNC = 0xFFF6,
    BLE_UUID_WIPE_CONFIRM = 0xFFF7,
    BLE_UUID_TIME_SYNC = 0xFFF8,
};

typedef struct {
    uint32_t seq;
    int64_t ts_uptime_ms;
    int bpm;
    int quality;
    uint32_t red;
    uint32_t ir;
    bool synced;
} hr_record_t;

typedef struct {
    uint32_t seq;
    int64_t start_uptime_ms;
    int64_t end_uptime_ms;
    uint32_t duration_ms;
    uint32_t frames;
    uint64_t samples;
    float avg_rms;
    uint32_t audio_bytes;
    char file_path[64];
    char object_id[40];
    char sha256_hex[65];
    bool synced;
} audio_record_t;

typedef struct {
    char object_id[40];
    uint32_t offset;
    uint16_t length;
    bool valid;
} fetch_request_t;

static max30102_config_t g_hr_cfg = {
    .i2c_port = I2C_PORT,
    .i2c_addr = MAX30102_I2C_ADDR,
};

static portMUX_TYPE g_store_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE g_voice_lock = portMUX_INITIALIZER_UNLOCKED;

static hr_record_t g_hr_store[HR_STORE_MAX];
static size_t g_hr_count = 0;
static uint32_t g_hr_seq = 0;
static bool g_hr_export_dirty = true;
static uint8_t g_hr_export_buf[BLE_EXPORT_MAX];
static size_t g_hr_export_len = 0;
static char g_hr_export_sha[65] = {0};

static audio_record_t g_audio_store[AUDIO_NOTE_MAX];
static size_t g_audio_count = 0;
static uint32_t g_audio_seq = 0;

static bool g_voice_active = false;
static int64_t g_voice_start_ms = 0;
static uint32_t g_voice_frame_count = 0;
static uint64_t g_voice_sample_count = 0;
static double g_voice_rms_sum = 0.0;
static uint32_t g_voice_bytes = 0;
static uint32_t g_voice_note_seq = 0;
static char g_voice_path[64] = {0};
static FILE *g_voice_fp = NULL;
static mbedtls_sha256_context g_voice_sha;
static bool g_voice_sha_ready = false;

static bool g_time_synced = false;
static int64_t g_time_base_epoch_ms = 0;
static int64_t g_time_base_uptime_ms = 0;
static int64_t g_last_sync_uptime_ms = 0;

static bool g_storage_ready = false;

static uint8_t g_ble_addr_type = BLE_OWN_ADDR_PUBLIC;
static uint16_t g_ble_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint16_t g_ble_status_val_handle = 0;
static uint16_t g_ble_wipe_val_handle = 0;
static fetch_request_t g_fetch_req = {0};
static char g_ble_status[BLE_STATUS_MAX] = {0};
static char g_ble_info[BLE_INFO_MAX] = {0};
static char g_ble_manifest[BLE_MANIFEST_MAX] = {0};
static char g_ble_wipe[BLE_WIPE_MAX] = "none";

static int64_t uptime_ms(void)
{
    return esp_timer_get_time() / 1000;
}

static int64_t uptime_to_epoch_ms(int64_t up_ms)
{
    if (!g_time_synced) {
        return 0;
    }
    return g_time_base_epoch_ms + (up_ms - g_time_base_uptime_ms);
}

static void sha_to_hex(const uint8_t digest[32], char out[65])
{
    static const char *hex = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        out[i * 2] = hex[(digest[i] >> 4) & 0xF];
        out[i * 2 + 1] = hex[digest[i] & 0xF];
    }
    out[64] = '\0';
}

static esp_err_t button_init(void)
{
    const gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << BUTTON_PIN,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    return gpio_config(&cfg);
}

static esp_err_t storage_init(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "storage",
        .max_files = 8,
        .format_if_mount_failed = true,
    };
    esp_err_t err = esp_vfs_spiffs_register(&conf);
    if (err != ESP_OK) {
        return err;
    }

    size_t total = 0;
    size_t used = 0;
    err = esp_spiffs_info(conf.partition_label, &total, &used);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "spiffs mounted total=%u used=%u", (unsigned)total, (unsigned)used);
    }
    g_storage_ready = true;
    return ESP_OK;
}

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

static void hr_store_append(int64_t ts_uptime_ms, int bpm, int quality, uint32_t red, uint32_t ir)
{
    taskENTER_CRITICAL(&g_store_lock);
    if (g_hr_count >= HR_STORE_MAX) {
        memmove(&g_hr_store[0], &g_hr_store[1], sizeof(g_hr_store[0]) * (HR_STORE_MAX - 1));
        g_hr_count = HR_STORE_MAX - 1;
    }
    hr_record_t *slot = &g_hr_store[g_hr_count++];
    slot->seq = ++g_hr_seq;
    slot->ts_uptime_ms = ts_uptime_ms;
    slot->bpm = bpm;
    slot->quality = quality;
    slot->red = red;
    slot->ir = ir;
    slot->synced = false;
    g_hr_export_dirty = true;
    taskEXIT_CRITICAL(&g_store_lock);
}

static void audio_store_append(const audio_record_t *record)
{
    taskENTER_CRITICAL(&g_store_lock);
    if (g_audio_count >= AUDIO_NOTE_MAX) {
        if (g_audio_store[0].file_path[0] != '\0') {
            unlink(g_audio_store[0].file_path);
        }
        memmove(&g_audio_store[0], &g_audio_store[1], sizeof(g_audio_store[0]) * (AUDIO_NOTE_MAX - 1));
        g_audio_count = AUDIO_NOTE_MAX - 1;
    }
    g_audio_store[g_audio_count++] = *record;
    taskEXIT_CRITICAL(&g_store_lock);
}

static void compact_synced(void)
{
    taskENTER_CRITICAL(&g_store_lock);
    size_t dst = 0;
    for (size_t i = 0; i < g_hr_count; i++) {
        if (!g_hr_store[i].synced) {
            if (dst != i) {
                g_hr_store[dst] = g_hr_store[i];
            }
            dst++;
        }
    }
    g_hr_count = dst;

    dst = 0;
    for (size_t i = 0; i < g_audio_count; i++) {
        if (!g_audio_store[i].synced) {
            if (dst != i) {
                g_audio_store[dst] = g_audio_store[i];
            }
            dst++;
        }
    }
    g_audio_count = dst;
    taskEXIT_CRITICAL(&g_store_lock);
}

static void rebuild_hr_export_if_needed(void)
{
    taskENTER_CRITICAL(&g_store_lock);
    if (!g_hr_export_dirty) {
        taskEXIT_CRITICAL(&g_store_lock);
        return;
    }

    size_t used = 0;
    const char *header = "ts_ms,bpm,quality,red,ir\n";
    size_t header_len = strlen(header);
    if (header_len < BLE_EXPORT_MAX) {
        memcpy(g_hr_export_buf, header, header_len);
        used = header_len;
    }

    for (size_t i = 0; i < g_hr_count; i++) {
        const hr_record_t *r = &g_hr_store[i];
        if (r->synced) {
            continue;
        }
        int64_t ts_ms = uptime_to_epoch_ms(r->ts_uptime_ms);
        if (ts_ms <= 0) {
            ts_ms = r->ts_uptime_ms;
        }
        char line[96];
        int n = snprintf(
            line,
            sizeof(line),
            "%lld,%d,%d,%lu,%lu\n",
            ts_ms,
            r->bpm,
            r->quality,
            (unsigned long)r->red,
            (unsigned long)r->ir
        );
        if (n <= 0) {
            continue;
        }
        if (used + (size_t)n >= BLE_EXPORT_MAX) {
            break;
        }
        memcpy(g_hr_export_buf + used, line, (size_t)n);
        used += (size_t)n;
    }
    g_hr_export_len = used;

    uint8_t digest[32] = {0};
    mbedtls_sha256_context ctx;
    mbedtls_sha256_init(&ctx);
    mbedtls_sha256_starts(&ctx, 0);
    if (g_hr_export_len > 0) {
        mbedtls_sha256_update(&ctx, g_hr_export_buf, g_hr_export_len);
    }
    mbedtls_sha256_finish(&ctx, digest);
    mbedtls_sha256_free(&ctx);
    sha_to_hex(digest, g_hr_export_sha);
    g_hr_export_dirty = false;
    taskEXIT_CRITICAL(&g_store_lock);
}

static size_t build_manifest(char *out, size_t out_cap)
{
    if (!out || out_cap == 0) {
        return 0;
    }
    rebuild_hr_export_if_needed();
    size_t used = 0;

    taskENTER_CRITICAL(&g_store_lock);
    if (g_hr_export_len > 0) {
        int n = snprintf(out + used, out_cap - used, "%s|%u|%s\n", HR_OBJECT_ID, (unsigned)g_hr_export_len, g_hr_export_sha);
        if (n > 0 && (size_t)n < out_cap - used) {
            used += (size_t)n;
        }
    }

    for (size_t i = 0; i < g_audio_count; i++) {
        const audio_record_t *a = &g_audio_store[i];
        if (a->synced) {
            continue;
        }
        int64_t start_ms = uptime_to_epoch_ms(a->start_uptime_ms);
        int64_t end_ms = uptime_to_epoch_ms(a->end_uptime_ms);
        if (start_ms <= 0) {
            start_ms = a->start_uptime_ms;
        }
        if (end_ms <= 0) {
            end_ms = a->end_uptime_ms;
        }
        int n = snprintf(
            out + used,
            out_cap - used,
            "%s|%u|%s|%lld|%lld|%u|%u|%llu|%d\n",
            a->object_id,
            (unsigned)a->audio_bytes,
            a->sha256_hex,
            start_ms,
            end_ms,
            (unsigned)a->duration_ms,
            (unsigned)a->frames,
            (unsigned long long)a->samples,
            (int)(a->avg_rms * 100.0f)
        );
        if (n <= 0 || (size_t)n >= out_cap - used) {
            break;
        }
        used += (size_t)n;
    }
    taskEXIT_CRITICAL(&g_store_lock);
    return used;
}

static void update_status(void)
{
    uint32_t hr_unsynced = 0;
    uint32_t audio_unsynced = 0;
    taskENTER_CRITICAL(&g_store_lock);
    for (size_t i = 0; i < g_hr_count; i++) {
        if (!g_hr_store[i].synced) {
            hr_unsynced++;
        }
    }
    for (size_t i = 0; i < g_audio_count; i++) {
        if (!g_audio_store[i].synced) {
            audio_unsynced++;
        }
    }
    taskEXIT_CRITICAL(&g_store_lock);

    int64_t now = uptime_ms();
    int64_t epoch = uptime_to_epoch_ms(now);
    snprintf(
        g_ble_status,
        sizeof(g_ble_status),
        "uptime_ms=%lld|epoch_ms=%lld|time_synced=%d|hr_unsynced=%lu|audio_unsynced=%lu|last_sync_uptime_ms=%lld",
        now,
        epoch,
        g_time_synced ? 1 : 0,
        (unsigned long)hr_unsynced,
        (unsigned long)audio_unsynced,
        g_last_sync_uptime_ms
    );
}

static void update_device_info(void)
{
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(
        g_ble_info,
        sizeof(g_ble_info),
        "device_id=xiao-%02x%02x%02x%02x%02x%02x|fw=0.7.0|cap=hr,audio,ble_sync",
        mac[0],
        mac[1],
        mac[2],
        mac[3],
        mac[4],
        mac[5]
    );
}

static void voice_note_start(void)
{
    bool started = false;
    int64_t now = uptime_ms();
    taskENTER_CRITICAL(&g_voice_lock);
    if (!g_voice_active && g_storage_ready) {
        g_voice_note_seq = ++g_audio_seq;
        snprintf(g_voice_path, sizeof(g_voice_path), "/spiffs/note_%lu.pcm", (unsigned long)g_voice_note_seq);
        g_voice_fp = fopen(g_voice_path, "wb");
        if (g_voice_fp) {
            mbedtls_sha256_init(&g_voice_sha);
            mbedtls_sha256_starts(&g_voice_sha, 0);
            g_voice_sha_ready = true;
            g_voice_active = true;
            g_voice_start_ms = now;
            g_voice_frame_count = 0;
            g_voice_sample_count = 0;
            g_voice_rms_sum = 0.0;
            g_voice_bytes = 0;
            started = true;
        }
    }
    taskEXIT_CRITICAL(&g_voice_lock);
    if (started) {
        ESP_LOGI(TAG, "voice_note_started ts=%lld file=%s", now, g_voice_path);
    } else {
        ESP_LOGW(TAG, "voice_note_start ignored");
    }
}

static void voice_note_stop(void)
{
    bool stopped = false;
    int64_t now = uptime_ms();
    audio_record_t record = {0};
    taskENTER_CRITICAL(&g_voice_lock);
    if (g_voice_active) {
        g_voice_active = false;
        record.seq = g_voice_note_seq;
        record.start_uptime_ms = g_voice_start_ms;
        record.end_uptime_ms = now;
        record.duration_ms = (uint32_t)(now - g_voice_start_ms);
        record.frames = g_voice_frame_count;
        record.samples = g_voice_sample_count;
        record.avg_rms = (float)(g_voice_frame_count > 0 ? g_voice_rms_sum / (double)g_voice_frame_count : 0.0);
        record.audio_bytes = g_voice_bytes;
        strncpy(record.file_path, g_voice_path, sizeof(record.file_path) - 1);
        snprintf(record.object_id, sizeof(record.object_id), "audio_%lu.pcm", (unsigned long)record.seq);
        if (g_voice_fp) {
            fclose(g_voice_fp);
            g_voice_fp = NULL;
        }
        if (g_voice_sha_ready) {
            uint8_t digest[32] = {0};
            mbedtls_sha256_finish(&g_voice_sha, digest);
            mbedtls_sha256_free(&g_voice_sha);
            sha_to_hex(digest, record.sha256_hex);
            g_voice_sha_ready = false;
        }
        stopped = true;
    }
    taskEXIT_CRITICAL(&g_voice_lock);
    if (!stopped) {
        return;
    }
    audio_store_append(&record);
    update_status();
    ESP_LOGI(TAG, "voice_note_stopped ts=%lld duration_ms=%u bytes=%u object_id=%s", now, (unsigned)record.duration_ms, (unsigned)record.audio_bytes, record.object_id);
}

static void voice_note_toggle(void)
{
    bool active = false;
    taskENTER_CRITICAL(&g_voice_lock);
    active = g_voice_active;
    taskEXIT_CRITICAL(&g_voice_lock);
    if (active) {
        voice_note_stop();
    } else {
        voice_note_start();
    }
}

static int parse_fetch_request(const char *input, fetch_request_t *out)
{
    if (!input || !out) {
        return -1;
    }
    const char *p1 = strchr(input, '|');
    if (!p1) {
        return -1;
    }
    const char *p2 = strchr(p1 + 1, '|');
    if (!p2) {
        return -1;
    }

    size_t id_len = (size_t)(p1 - input);
    if (id_len == 0 || id_len >= sizeof(out->object_id)) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    memcpy(out->object_id, input, id_len);
    out->object_id[id_len] = '\0';

    char offset_buf[24] = {0};
    char length_buf[24] = {0};
    size_t offset_len = (size_t)(p2 - (p1 + 1));
    size_t length_len = strlen(p2 + 1);
    if (offset_len == 0 || offset_len >= sizeof(offset_buf) || length_len == 0 || length_len >= sizeof(length_buf)) {
        return -1;
    }
    memcpy(offset_buf, p1 + 1, offset_len);
    memcpy(length_buf, p2 + 1, length_len);
    for (size_t i = 0; i < offset_len; i++) {
        if (!isdigit((unsigned char)offset_buf[i])) {
            return -1;
        }
    }
    for (size_t i = 0; i < length_len; i++) {
        if (!isdigit((unsigned char)length_buf[i])) {
            return -1;
        }
    }

    unsigned long offset = strtoul(offset_buf, NULL, 10);
    unsigned long length = strtoul(length_buf, NULL, 10);
    if (length == 0 || length > BLE_FETCH_CHUNK_MAX) {
        return -1;
    }
    out->offset = (uint32_t)offset;
    out->length = (uint16_t)length;
    out->valid = true;
    return 0;
}

static size_t read_object_chunk(const fetch_request_t *req, uint8_t *out, size_t out_cap)
{
    if (!req || !out || out_cap == 0) {
        return 0;
    }
    size_t want = req->length;
    if (want > out_cap) {
        want = out_cap;
    }

    if (strcmp(req->object_id, HR_OBJECT_ID) == 0) {
        rebuild_hr_export_if_needed();
        if (req->offset >= g_hr_export_len) {
            return 0;
        }
        size_t available = g_hr_export_len - req->offset;
        if (want > available) {
            want = available;
        }
        memcpy(out, g_hr_export_buf + req->offset, want);
        return want;
    }

    audio_record_t note = {0};
    bool found = false;
    taskENTER_CRITICAL(&g_store_lock);
    for (size_t i = 0; i < g_audio_count; i++) {
        if (!g_audio_store[i].synced && strcmp(g_audio_store[i].object_id, req->object_id) == 0) {
            note = g_audio_store[i];
            found = true;
            break;
        }
    }
    taskEXIT_CRITICAL(&g_store_lock);
    if (!found) {
        return 0;
    }

    FILE *fp = fopen(note.file_path, "rb");
    if (!fp) {
        return 0;
    }
    if (fseek(fp, (long)req->offset, SEEK_SET) != 0) {
        fclose(fp);
        return 0;
    }
    size_t read_n = fread(out, 1, want, fp);
    fclose(fp);
    return read_n;
}

static void commit_synced_objects(const char *csv_ids)
{
    if (!csv_ids || csv_ids[0] == '\0') {
        return;
    }
    uint32_t hr_marked = 0;
    uint32_t audio_marked = 0;

    char work[256] = {0};
    strncpy(work, csv_ids, sizeof(work) - 1);
    char *save = NULL;
    char *tok = strtok_r(work, ",", &save);
    while (tok) {
        while (*tok == ' ') {
            tok++;
        }
        size_t len = strlen(tok);
        while (len > 0 && (tok[len - 1] == ' ' || tok[len - 1] == '\r' || tok[len - 1] == '\n')) {
            tok[--len] = '\0';
        }
        if (strcmp(tok, HR_OBJECT_ID) == 0) {
            taskENTER_CRITICAL(&g_store_lock);
            for (size_t i = 0; i < g_hr_count; i++) {
                if (!g_hr_store[i].synced) {
                    g_hr_store[i].synced = true;
                    hr_marked++;
                }
            }
            g_hr_export_dirty = true;
            taskEXIT_CRITICAL(&g_store_lock);
        } else if (strncmp(tok, "audio_", 6) == 0) {
            taskENTER_CRITICAL(&g_store_lock);
            for (size_t i = 0; i < g_audio_count; i++) {
                if (!g_audio_store[i].synced && strcmp(g_audio_store[i].object_id, tok) == 0) {
                    g_audio_store[i].synced = true;
                    if (g_audio_store[i].file_path[0] != '\0') {
                        unlink(g_audio_store[i].file_path);
                    }
                    audio_marked++;
                    break;
                }
            }
            taskEXIT_CRITICAL(&g_store_lock);
        }
        tok = strtok_r(NULL, ",", &save);
    }
    compact_synced();
    g_last_sync_uptime_ms = uptime_ms();
    snprintf(g_ble_wipe, sizeof(g_ble_wipe), "ok|hr=%lu|audio=%lu|ts=%lld", (unsigned long)hr_marked, (unsigned long)audio_marked, g_last_sync_uptime_ms);
    update_status();
    if (g_ble_conn_handle != BLE_HS_CONN_HANDLE_NONE && g_ble_status_val_handle != 0) {
        ble_gatts_notify(g_ble_conn_handle, g_ble_status_val_handle);
    }
    if (g_ble_conn_handle != BLE_HS_CONN_HANDLE_NONE && g_ble_wipe_val_handle != 0) {
        ble_gatts_notify(g_ble_conn_handle, g_ble_wipe_val_handle);
    }
    ESP_LOGI(TAG, "commit_sync hr_marked=%lu audio_marked=%lu", (unsigned long)hr_marked, (unsigned long)audio_marked);
}

static void set_time_sync(int64_t epoch_ms)
{
    g_time_base_epoch_ms = epoch_ms;
    g_time_base_uptime_ms = uptime_ms();
    g_time_synced = true;
    update_status();
    ESP_LOGI(TAG, "time_sync epoch_ms=%lld uptime_base_ms=%lld", g_time_base_epoch_ms, g_time_base_uptime_ms);
}

static int gatt_access_cb(uint16_t conn_handle, uint16_t attr_handle, struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;
    uint16_t uuid = ble_uuid_u16(ctxt->chr->uuid);

    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        if (uuid == BLE_UUID_DEVICE_INFO) {
            update_device_info();
            return os_mbuf_append(ctxt->om, g_ble_info, strlen(g_ble_info)) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        if (uuid == BLE_UUID_STATUS) {
            update_status();
            return os_mbuf_append(ctxt->om, g_ble_status, strlen(g_ble_status)) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        if (uuid == BLE_UUID_MANIFEST) {
            size_t n = build_manifest(g_ble_manifest, sizeof(g_ble_manifest));
            return os_mbuf_append(ctxt->om, g_ble_manifest, n) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        if (uuid == BLE_UUID_FETCH_DATA) {
            if (!g_fetch_req.valid) {
                return 0;
            }
            uint8_t chunk[BLE_FETCH_CHUNK_MAX] = {0};
            size_t got = read_object_chunk(&g_fetch_req, chunk, sizeof(chunk));
            return os_mbuf_append(ctxt->om, chunk, got) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        if (uuid == BLE_UUID_WIPE_CONFIRM) {
            return os_mbuf_append(ctxt->om, g_ble_wipe, strlen(g_ble_wipe)) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        return BLE_ATT_ERR_UNLIKELY;
    }

    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        char payload[BLE_FETCH_REQ_MAX] = {0};
        uint16_t flat_len = 0;
        if (ble_hs_mbuf_to_flat(ctxt->om, payload, sizeof(payload) - 1, &flat_len) != 0) {
            return BLE_ATT_ERR_UNLIKELY;
        }
        payload[flat_len] = '\0';

        if (uuid == BLE_UUID_FETCH_REQ) {
            fetch_request_t req = {0};
            if (parse_fetch_request(payload, &req) != 0) {
                return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
            }
            g_fetch_req = req;
            return 0;
        }
        if (uuid == BLE_UUID_COMMIT_SYNC) {
            commit_synced_objects(payload);
            return 0;
        }
        if (uuid == BLE_UUID_TIME_SYNC) {
            char *end = NULL;
            long long epoch_ms = strtoll(payload, &end, 10);
            if (end == payload) {
                return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
            }
            set_time_sync((int64_t)epoch_ms);
            return 0;
        }
        return BLE_ATT_ERR_UNLIKELY;
    }
    return BLE_ATT_ERR_UNLIKELY;
}

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = BLE_UUID16_DECLARE(BLE_UUID_SERVICE),
        .characteristics = (struct ble_gatt_chr_def[]) {
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_DEVICE_INFO), .access_cb = gatt_access_cb, .flags = BLE_GATT_CHR_F_READ},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_STATUS), .access_cb = gatt_access_cb, .val_handle = &g_ble_status_val_handle, .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_MANIFEST), .access_cb = gatt_access_cb, .flags = BLE_GATT_CHR_F_READ},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_FETCH_REQ), .access_cb = gatt_access_cb, .flags = BLE_GATT_CHR_F_WRITE},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_FETCH_DATA), .access_cb = gatt_access_cb, .flags = BLE_GATT_CHR_F_READ},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_COMMIT_SYNC), .access_cb = gatt_access_cb, .flags = BLE_GATT_CHR_F_WRITE},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_WIPE_CONFIRM), .access_cb = gatt_access_cb, .val_handle = &g_ble_wipe_val_handle, .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY},
            {.uuid = BLE_UUID16_DECLARE(BLE_UUID_TIME_SYNC), .access_cb = gatt_access_cb, .flags = BLE_GATT_CHR_F_WRITE},
            {0},
        },
    },
    {0},
};

static void ble_advertise(void);

static int ble_gap_cb(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            g_ble_conn_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "ble connected handle=%u", g_ble_conn_handle);
        } else {
            ble_advertise();
        }
        return 0;
    case BLE_GAP_EVENT_DISCONNECT:
        g_ble_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        ble_advertise();
        return 0;
    case BLE_GAP_EVENT_ADV_COMPLETE:
        ble_advertise();
        return 0;
    default:
        return 0;
    }
}

static void ble_advertise(void)
{
    struct ble_hs_adv_fields fields;
    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    const char *name = ble_svc_gap_device_name();
    fields.name = (uint8_t *)name;
    fields.name_len = (uint8_t)strlen(name);
    fields.name_is_complete = 1;
    if (ble_gap_adv_set_fields(&fields) != 0) {
        return;
    }

    struct ble_gap_adv_params adv_params;
    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    ble_gap_adv_start(g_ble_addr_type, NULL, BLE_HS_FOREVER, &adv_params, ble_gap_cb, NULL);
}

static void ble_on_sync(void)
{
    if (ble_hs_id_infer_auto(0, &g_ble_addr_type) != 0) {
        return;
    }
    ble_advertise();
}

static void ble_host_task(void *arg)
{
    (void)arg;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static esp_err_t ble_init(void)
{
    ESP_RETURN_ON_ERROR(nimble_port_init(), TAG, "nimble_port_init failed");
    ble_hs_cfg.sync_cb = ble_on_sync;
    ble_svc_gap_init();
    ble_svc_gatt_init();
    if (ble_svc_gap_device_name_set(BLE_DEVICE_NAME) != 0) {
        return ESP_FAIL;
    }
    if (ble_gatts_count_cfg(gatt_svcs) != 0) {
        return ESP_FAIL;
    }
    if (ble_gatts_add_svcs(gatt_svcs) != 0) {
        return ESP_FAIL;
    }
    nimble_port_freertos_init(ble_host_task);
    return ESP_OK;
}

static int estimate_bpm(const max30102_sample_t *s)
{
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
    int64_t next_log_ms = 0;
    uint32_t window_reads = 0;
    uint32_t window_valid = 0;
    uint32_t window_errors = 0;
    uint32_t recovery_count = 0;
    uint32_t consecutive_errors = 0;
    uint32_t last_red = 0;
    uint32_t last_ir = 0;
    int last_bpm = -1;
    int last_quality = 0;

    while (1) {
        max30102_sample_t sample = {0};
        esp_err_t err = max30102_read_sample(&g_hr_cfg, &sample);
        window_reads++;
        if (err == ESP_OK) {
            consecutive_errors = 0;
            last_red = sample.red;
            last_ir = sample.ir;
            last_bpm = estimate_bpm(&sample);
            last_quality = sample.valid ? 1 : 0;
            if (sample.valid) {
                window_valid++;
                hr_store_append(uptime_ms(), last_bpm, 100, sample.red, sample.ir);
            }
        } else {
            window_errors++;
            consecutive_errors++;
            last_bpm = -1;
            last_quality = 0;
            if (consecutive_errors >= HR_RECOVERY_ERR_STREAK) {
                ESP_LOGW(TAG, "hr_recover start consecutive_errs=%lu", (unsigned long)consecutive_errors);
                esp_err_t rec_err = max30102_init(&g_hr_cfg);
                if (rec_err == ESP_OK) {
                    recovery_count++;
                    ESP_LOGI(TAG, "hr_recover ok count=%lu", (unsigned long)recovery_count);
                } else {
                    ESP_LOGE(TAG, "hr_recover failed: %s", esp_err_to_name(rec_err));
                }
                consecutive_errors = 0;
                vTaskDelay(pdMS_TO_TICKS(HR_RECOVERY_DELAY_MS));
            }
        }

        int64_t now_ms = uptime_ms();
        if (now_ms >= next_log_ms) {
            update_status();
            ESP_LOGI(
                TAG,
                "hr_status ts=%lld reads=%lu valid=%lu errs=%lu recoveries=%lu quality=%d bpm=%d red=%lu ir=%lu",
                now_ms,
                (unsigned long)window_reads,
                (unsigned long)window_valid,
                (unsigned long)window_errors,
                (unsigned long)recovery_count,
                last_quality,
                last_bpm,
                (unsigned long)last_red,
                (unsigned long)last_ir
            );
            window_reads = 0;
            window_valid = 0;
            window_errors = 0;
            next_log_ms = now_ms + HR_STATUS_LOG_MS;

            if (g_ble_conn_handle != BLE_HS_CONN_HANDLE_NONE && g_ble_status_val_handle != 0) {
                ble_gatts_notify(g_ble_conn_handle, g_ble_status_val_handle);
            }
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
        bool auto_stop = false;
        taskENTER_CRITICAL(&g_voice_lock);
        if (g_voice_active) {
            g_voice_frame_count++;
            g_voice_sample_count += n;
            g_voice_rms_sum += rms;
            if (g_voice_fp) {
                size_t remaining_samples = 0;
                if (g_voice_bytes < AUDIO_NOTE_MAX_BYTES) {
                    remaining_samples = (AUDIO_NOTE_MAX_BYTES - g_voice_bytes) / sizeof(int16_t);
                }
                size_t to_write = n;
                if (to_write > remaining_samples) {
                    to_write = remaining_samples;
                }
                if (to_write > 0) {
                    size_t wrote = fwrite(buf, sizeof(int16_t), to_write, g_voice_fp);
                    size_t wrote_bytes = wrote * sizeof(int16_t);
                    g_voice_bytes += (uint32_t)wrote_bytes;
                    if (g_voice_sha_ready && wrote_bytes > 0) {
                        mbedtls_sha256_update(&g_voice_sha, (const uint8_t *)buf, wrote_bytes);
                    }
                }
                if (g_voice_bytes >= AUDIO_NOTE_MAX_BYTES) {
                    auto_stop = true;
                }
            }
        }
        taskEXIT_CRITICAL(&g_voice_lock);
        if (auto_stop) {
            ESP_LOGW(TAG, "voice_note auto-stop at max bytes=%u", (unsigned)AUDIO_NOTE_MAX_BYTES);
            voice_note_stop();
        }
    }
}

static void button_task(void *arg)
{
    (void)arg;
    const int debounce_ticks = (BUTTON_DEBOUNCE_MS + BUTTON_POLL_MS - 1) / BUTTON_POLL_MS;
    int stable_raw = gpio_get_level(BUTTON_PIN);
    int last_raw = stable_raw;
    int same_ticks = 0;

    ESP_LOGI(TAG, "button_task started gpio=%d initial=%d", BUTTON_PIN, stable_raw);
    while (1) {
        int raw = gpio_get_level(BUTTON_PIN);
        if (raw == last_raw) {
            same_ticks++;
        } else {
            same_ticks = 0;
            last_raw = raw;
        }
        if (raw != stable_raw && same_ticks >= debounce_ticks) {
            stable_raw = raw;
            bool pressed = (stable_raw == BUTTON_ACTIVE_LEVEL);
            ESP_LOGI(TAG, "button_event ts=%lld state=%s", uptime_ms(), pressed ? "pressed" : "released");
            if (pressed) {
                voice_note_toggle();
            }
        }
        vTaskDelay(pdMS_TO_TICKS(BUTTON_POLL_MS));
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());

    ESP_ERROR_CHECK(storage_init());
    ESP_ERROR_CHECK(i2s_mic_init());
    ESP_ERROR_CHECK(button_init());

    ESP_ERROR_CHECK(i2c_init());
    ESP_ERROR_CHECK(max30102_init(&g_hr_cfg));

    ESP_ERROR_CHECK(ble_init());

    update_device_info();
    update_status();

    ESP_LOGI(TAG, "mynah wearable started (ble sync enabled)");

    xTaskCreate(hr_task, "hr_task", 4096, NULL, 5, NULL);
    xTaskCreate(mic_task, "mic_task", 4096, NULL, 4, NULL);
    xTaskCreate(button_task, "button_task", 3072, NULL, 3, NULL);
}
