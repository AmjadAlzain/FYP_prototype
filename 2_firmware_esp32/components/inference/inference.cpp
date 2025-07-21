// components/inference/inference.cpp
#include "inference.h"
#include "model_data.h"

#include <cstring>
#include <stdint.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"

#include "esp_heap_caps.h"

// ---------- arenas ----------
#define TFLITE_DET_ARENA (460 * 1024)
#define TFLITE_FEAT_ARENA (50 * 1024)

// ---------- globals ----------
namespace {
    tflite::MicroMutableOpResolver<20> resolver;

    const tflite::Model* detector_model  = nullptr;
    tflite::MicroInterpreter* det_interp = nullptr;
    uint8_t* det_arena = nullptr;
    TfLiteTensor* det_in  = nullptr;
    TfLiteTensor* det_out = nullptr;

    const tflite::Model* feature_model = nullptr;
    tflite::MicroInterpreter* feat_interp = nullptr;
    uint8_t* feat_arena = nullptr;
    TfLiteTensor* feat_in  = nullptr;
    TfLiteTensor* feat_out = nullptr;
}  // namespace



/*------------------------------------------------------------------*/
static esp_err_t register_required_ops()
{
    // Convolution (INT8 kernels are picked automatically for a quant model)
    if (resolver.AddConv2D()            != kTfLiteOk) return ESP_FAIL;
    if (resolver.AddDepthwiseConv2D()   != kTfLiteOk) return ESP_FAIL;

    // Pool / pad / transpose
    if (resolver.AddMaxPool2D()         != kTfLiteOk) return ESP_FAIL;
    if (resolver.AddPad()               != kTfLiteOk) return ESP_FAIL;
    if (resolver.AddTranspose()         != kTfLiteOk) return ESP_FAIL;

    // Shape ops
    if (resolver.AddReshape()           != kTfLiteOk) return ESP_FAIL;
    if (resolver.AddResizeNearestNeighbor() != kTfLiteOk) return ESP_FAIL;

    // Element-wise math
    if (resolver.AddAdd()               != kTfLiteOk) return ESP_FAIL;
    if (resolver.AddMul()               != kTfLiteOk) return ESP_FAIL;
    if (resolver.AddLogistic()          != kTfLiteOk) return ESP_FAIL;

    return ESP_OK;
}



/*------------------------------------------------------------------*/
static void preprocess_resize_crop_normalize(const camera_fb_t* fb,
                                             int x, int y, int w, int h,
                                             int out_w, int out_h,
                                             int8_t* out_buf)
{
    static uint16_t tmp[DETECTOR_INPUT_WIDTH * DETECTOR_INPUT_HEIGHT];

    for (int j = 0; j < out_h; ++j) {
        int sj = y + (j * h) / out_h;
        for (int i = 0; i < out_w; ++i) {
            int si = x + (i * w) / out_w;
            tmp[j * out_w + i] =
                reinterpret_cast<const uint16_t*>(fb->buf)[sj * fb->width + si];
        }
    }

    int idx = 0;
    for (int j = 0; j < out_h; ++j) {
        for (int i = 0; i < out_w; ++i) {
            uint16_t p = tmp[j * out_w + i];
            uint8_t r = ((p >> 11) & 0x1F) << 3;
            uint8_t g = ((p >>  5) & 0x3F) << 2;
            uint8_t b =  (p        & 0x1F) << 3;
            out_buf[idx++] = static_cast<int8_t>(r - 128);
            out_buf[idx++] = static_cast<int8_t>(g - 128);
            out_buf[idx++] = static_cast<int8_t>(b - 128);
        }
    }
}



/*------------------------------------------------------------------*/
esp_err_t inference_init(void)
{
    det_arena  = (uint8_t*)heap_caps_malloc(TFLITE_DET_ARENA,  MALLOC_CAP_SPIRAM);
    feat_arena = (uint8_t*)heap_caps_malloc(TFLITE_FEAT_ARENA, MALLOC_CAP_SPIRAM);
    if (!det_arena || !feat_arena) return ESP_FAIL;

    size_t len;
    detector_model = tflite::GetModel(get_detector_model_tflite(&len));
    feature_model  = tflite::GetModel(get_feature_extractor_model_tflite(&len));

    if (detector_model->version() != TFLITE_SCHEMA_VERSION ||
        feature_model ->version() != TFLITE_SCHEMA_VERSION) {
        return ESP_FAIL;
    }

    if (register_required_ops() != ESP_OK) return ESP_FAIL;

    static tflite::MicroInterpreter det_i(detector_model, resolver,
                                          det_arena, TFLITE_DET_ARENA);
    det_interp = &det_i;
    if (det_interp->AllocateTensors() != kTfLiteOk) return ESP_FAIL;
    det_in  = det_interp->input(0);
    det_out = det_interp->output(0);

    static tflite::MicroInterpreter feat_i(feature_model, resolver,
                                           feat_arena, TFLITE_FEAT_ARENA);
    feat_interp = &feat_i;
    if (feat_interp->AllocateTensors() != kTfLiteOk) return ESP_FAIL;
    feat_in  = feat_interp->input(0);
    feat_out = feat_interp->output(0);

    hdc_init_prototypes();
    return ESP_OK;
}



/*------------------------------------------------------------------*/
esp_err_t inference_detect(const camera_fb_t* fb, bbox_t* box)
{
    static int8_t det_buf[DETECTOR_INPUT_WIDTH * DETECTOR_INPUT_HEIGHT * 3];
    preprocess_resize_crop_normalize(fb, 0, 0, fb->width, fb->height,
                                     DETECTOR_INPUT_WIDTH, DETECTOR_INPUT_HEIGHT,
                                     det_buf);

    memcpy(det_in->data.int8, det_buf, sizeof(det_buf));
    if (det_interp->Invoke() != kTfLiteOk) return ESP_FAIL;

    const float* o = det_out->data.f;   // [score,cx,cy,w,h]
    box->score = o[0];
    float cx=o[1], cy=o[2], w=o[3], h=o[4];
    box->x = (int)((cx - w/2.f) * fb->width);
    box->y = (int)((cy - h/2.f) * fb->height);
    box->w = (int)(w * fb->width);
    box->h = (int)(h * fb->height);
    return ESP_OK;
}



/*------------------------------------------------------------------*/
esp_err_t inference_extract_features(const camera_fb_t* fb,
                                     const bbox_t* roi,
                                     int8_t* feature_vec)
{
    static int8_t feat_buf[FEATURE_INPUT_WIDTH * FEATURE_INPUT_HEIGHT * 3];
    preprocess_resize_crop_normalize(fb,
                                     roi->x, roi->y, roi->w, roi->h,
                                     FEATURE_INPUT_WIDTH, FEATURE_INPUT_HEIGHT,
                                     feat_buf);

    memcpy(feat_in->data.int8, feat_buf, sizeof(feat_buf));
    if (feat_interp->Invoke() != kTfLiteOk) return ESP_FAIL;

    memcpy(feature_vec, feat_out->data.int8, HDC_INPUT_DIM);
    return ESP_OK;
}
