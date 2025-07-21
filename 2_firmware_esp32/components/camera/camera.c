// camera/camera.c
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp32_s3_eye.h"    // now resolves, BSP include path is public
#include "esp_camera_driver.h"        // your shim: esp_camera_driver API
#include "camera.h"

static const char *TAG = "camera";

static void camera_capture_task(void *arg)
{
    QueueHandle_t frameQ = (QueueHandle_t)arg;
    while (1) {
        camera_fb_t *fb = camera_driver_grab();  // from your camera_driver component
        if (!fb) {
            ESP_LOGE(TAG, "grab failed");
            continue;
        }
        if (xQueueSend(frameQ, &fb, portMAX_DELAY) != pdTRUE) {
            camera_driver_return(fb);
        }
    }
}

esp_err_t register_camera(pixformat_t pixformat,
                          framesize_t framesize,
                          size_t fb_count,
                          QueueHandle_t out_queue)
{
    camera_config_t cfg = BSP_CAMERA_DEFAULT_CONFIG;
    cfg.pixel_format = pixformat;
    cfg.frame_size   = framesize;
    cfg.fb_count     = fb_count;
    esp_err_t err = camera_driver_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "init failed: %d", err);
        return err;
    }
    xTaskCreatePinnedToCore(camera_capture_task, "cam_task",
                            4*1024, out_queue, 5, NULL, 1);
    return ESP_OK;
}
