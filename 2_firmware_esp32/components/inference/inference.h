#pragma once

#include "esp_err.h"
#include "esp_camera.h"
#include "hdc.h" // <-- This single include replaces the old, incorrect defines.

#ifdef __cplusplus
extern "C" {
#endif

// Model input sizes, must match your TFLite model's expected input
#define DETECTOR_INPUT_WIDTH    160
#define DETECTOR_INPUT_HEIGHT   160
#define FEATURE_INPUT_WIDTH     64
#define FEATURE_INPUT_HEIGHT    64

// Note: HDC_INPUT_DIM, HDC_FEATURE_DIM, and NUM_CLASSES are now
// correctly included from hdc.h -> hdc_config.h

// A struct to hold bounding box coordinates and score
typedef struct {
    int x;
    int y;
    int w;
    int h;
    float score;
} bbox_t;

/**
 * @brief Initializes TFLite interpreters and the HDC model.
 */
esp_err_t inference_init(void);

/**
 * @brief Takes a raw camera frame, runs a detector, and returns the best bounding box.
 */
esp_err_t inference_detect(const camera_fb_t *fb, bbox_t *box_out);

/**
 * @brief Takes a camera frame + ROI, and runs the feature extractor model.
 */
esp_err_t inference_extract_features(const camera_fb_t *full_fb,
                                     const bbox_t *roi,
                                     int8_t *feature_out);

// The 'inference_classify' function is no longer needed here.
// You should call 'hdc_classify' directly from your main application logic.

// The 'extern int8_t hdc_prototypes' declaration is also no longer needed here,
// as it is correctly exposed in 'hdc.h'.

#ifdef __cplusplus
}
#endif