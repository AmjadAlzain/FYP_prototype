// camera/camera.h

#ifndef CAMERA_CAMERA_H
#define CAMERA_CAMERA_H

#include "esp_camera.h"   // brings pixformat_t, framesize_t, camera_fb_t
#include "esp_err.h"
#include "freertos/FreeRTOS.h"

esp_err_t register_camera(pixformat_t pixformat,
                          framesize_t framesize,
                          size_t fb_count,
                          QueueHandle_t out_queue);

#endif // CAMERA_CAMERA_H
