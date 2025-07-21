// components/camera_driver/esp_camera_driver.c
#include "driver/gpio.h"  
#include "esp_log.h"
#include "esp_camera_driver.h"

static const char *TAG = "cam_driver";

esp_err_t camera_driver_init(const camera_config_t *config)
{
    /* make sure SCCB lines have pull-ups — common cause of
       “i2c.master: probe device timeout” / 0x105 errors               */
    gpio_set_pull_mode(config->pin_sccb_sda, GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(config->pin_sccb_scl, GPIO_PULLUP_ONLY);
    ESP_LOGI(TAG, "Initializing esp_camera...");
    return esp_camera_init(config);
}

camera_fb_t *camera_driver_grab()
{
    return esp_camera_fb_get();
}

void camera_driver_return(camera_fb_t *fb)
{
    esp_camera_fb_return(fb);
}
