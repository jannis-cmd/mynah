idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_timer nvs_flash max30102 bt spiffs mbedtls
)
