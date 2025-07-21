// components/camera/camera_setup.h
#pragma once

#include "esp_camera.h"

/* ─── ESP-S3-EYE pinout ────────────────────────────────────────── */
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1

#define CAM_PIN_XCLK   15      // was 40  -- fixed
#define CAM_PIN_PCLK   13      // was 39
#define CAM_PIN_VSYNC  6       // was 41
#define CAM_PIN_HREF   7       // was 42

#define CAM_PIN_SIOD   4       // was 17
#define CAM_PIN_SIOC   5       // was 18

#define CAM_PIN_D0     11
#define CAM_PIN_D1     9
#define CAM_PIN_D2     8
#define CAM_PIN_D3     10
#define CAM_PIN_D4     12
#define CAM_PIN_D5     18      // was 14
#define CAM_PIN_D6     17      // was 21
#define CAM_PIN_D7     16      // was 47



/* ─── Image & sensor defaults ──────────────────────────────────── */
#define XCLK_FREQ_HZ        15000000        // 15 MHz is the value Espressif use
#define FRAME_SIZE          FRAMESIZE_QVGA  // 320×240
#define PIXFORMAT           PIXFORMAT_RGB565

#define FRAME_WIDTH         320
#define FRAME_HEIGHT        240

// Model input sizes
#define DETECTOR_INPUT_WIDTH    160
#define DETECTOR_INPUT_HEIGHT   160
#define FEATURE_INPUT_WIDTH     64
#define FEATURE_INPUT_HEIGHT    64

// ImageNet mean/std for normalization
static const float IMG_MEAN[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
static const float IMG_STD[3]  = {0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f};

// Camera init and helpers
esp_err_t camera_init();
camera_fb_t* capture_frame();

// Preprocessing: resize/crop/normalize
void resize_crop_image(const uint8_t* src_buf, int w_src, int h_src,
                       uint8_t* dest_buf, int w_dst, int h_dst,
                       int roi_x, int roi_y, int roi_w, int roi_h);

void normalize_image(uint8_t* image, int pixel_count);
