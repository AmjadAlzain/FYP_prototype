// components/camera/camera_setup.c
#include "camera_setup.h"
#include "driver/gpio.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

esp_err_t camera_init() {
    camera_config_t config = {
        .pin_pwdn     = CAM_PIN_PWDN,
        .pin_reset    = CAM_PIN_RESET,
        .pin_xclk     = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7       = CAM_PIN_D7,
        .pin_d6       = CAM_PIN_D6,
        .pin_d5       = CAM_PIN_D5,
        .pin_d4       = CAM_PIN_D4,
        .pin_d3       = CAM_PIN_D3,
        .pin_d2       = CAM_PIN_D2,
        .pin_d1       = CAM_PIN_D1,
        .pin_d0       = CAM_PIN_D0,
        .pin_vsync    = CAM_PIN_VSYNC,
        .pin_href     = CAM_PIN_HREF,
        .pin_pclk     = CAM_PIN_PCLK,

        .xclk_freq_hz = XCLK_FREQ_HZ,       // 15 MHz
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        .pixel_format = PIXFORMAT,
        .frame_size   = FRAME_SIZE,
        .jpeg_quality = 12,
        .fb_count     = 2,                   // two buffers help if PSRAM is present
#if CONFIG_SPIRAM
        .fb_location  = CAMERA_FB_IN_PSRAM,
#endif
        .grab_mode    = CAMERA_GRAB_WHEN_EMPTY,
    };

    /* guarantee internal pull-ups on SCCB */
    gpio_set_pull_mode(CAM_PIN_SIOD, GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(CAM_PIN_SIOC, GPIO_PULLUP_ONLY);

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        printf("Camera init failed: 0x%x\n", err);
        return err;
    }

    /* Optional sensor tweaks – flip if the image is upside-down */
    sensor_t *s = esp_camera_sensor_get();
    if (s && s->id.PID == OV2640_PID) {
        s->set_vflip(s, 1);
    }
    return ESP_OK;
}

camera_fb_t* capture_frame() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        printf("Camera capture failed\n");
        return NULL;
    }
    return fb;
}

// Resize + crop, nearest neighbor. Handles RGB565.
void resize_crop_image(const uint8_t* src_buf, int w_src, int h_src,
                       uint8_t* dst_buf, int w_dst, int h_dst,
                       int roi_x, int roi_y, int roi_w, int roi_h) {
    if (roi_w == 0 || roi_h == 0) {
        roi_x = roi_y = 0;
        roi_w = w_src;
        roi_h = h_src;
    }
    for (int j = 0; j < h_dst; ++j) {
        int src_j = roi_y + (j * roi_h) / h_dst;
        for (int i = 0; i < w_dst; ++i) {
            int src_i = roi_x + (i * roi_w) / w_dst;
            const uint16_t* src_pixels = (const uint16_t*)src_buf;
            uint16_t* dst_pixels = (uint16_t*)dst_buf;
            dst_pixels[j * w_dst + i] = src_pixels[src_j * w_src + src_i];
        }
    }
}

// Simplified normalization: treat each byte as channel (approx for RGB565)
void normalize_image(uint8_t* image, int pixel_count) {
    for (int idx = 0; idx < pixel_count; idx += 3) {
        float r = image[idx + 0];
        float g = image[idx + 1];
        float b = image[idx + 2];
        r = (r - IMG_MEAN[0]) / IMG_STD[0];
        g = (g - IMG_MEAN[1]) / IMG_STD[1];
        b = (b - IMG_MEAN[2]) / IMG_STD[2];
        int8_t qr = (int8_t) (r * 128);
        int8_t qg = (int8_t) (g * 128);
        int8_t qb = (int8_t) (b * 128);
        image[idx + 0] = *(uint8_t*)&qr;
        image[idx + 1] = *(uint8_t*)&qg;
        image[idx + 2] = *(uint8_t*)&qb;
    }
}
