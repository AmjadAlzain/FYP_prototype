// board/esp32_s3_eye.c (minimal wrapper)
#include "esp_log.h"
#include "esp32_s3_eye.h"

static const char *TAG = "board";

esp_err_t board_init(void)
{
    ESP_LOGI(TAG, "Initializing board peripherals...");

    // Initialize I2C (for camera SCCB, IMU, etc.)
    ESP_ERROR_CHECK(bsp_i2c_init());   // Initialize I2C bus:contentReference[oaicite:14]{index=14}

    // Initialize and start LCD (with default config)
    bsp_display_start();              // Initialize display (LVGL):contentReference[oaicite:15]{index=15}
    bsp_display_brightness_init();    // Set up PWM for backlight control
    bsp_display_backlight_on();       // Turn on LCD backlight:contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}

    ESP_LOGI(TAG, "Board init complete");
    return ESP_OK;
}
