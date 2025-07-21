// components/camera_driver/esp_camera_driver.h
#ifndef ESP_CAMERA_DRIVER_H
#define ESP_CAMERA_DRIVER_H

#include "esp_err.h"
#include "esp_camera.h"

// a tiny wrapper, if you like:
// can put board‐specific config defaults here
esp_err_t camera_driver_init(const camera_config_t *config);
camera_fb_t *camera_driver_grab();
void camera_driver_return(camera_fb_t *fb);

#endif // ESP_CAMERA_DRIVER_H
