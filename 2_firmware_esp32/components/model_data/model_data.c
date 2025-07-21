// components/model_data/model_data.c
#include "model_data.h"

extern const unsigned char _binary_detector_model_tflite_start[]   asm("_binary_detector_model_tflite_start");
extern const unsigned char _binary_detector_model_tflite_end[]     asm("_binary_detector_model_tflite_end");
extern const unsigned char _binary_feature_extractor_model_tflite_start[] asm("_binary_feature_extractor_model_tflite_start");
extern const unsigned char _binary_feature_extractor_model_tflite_end[]   asm("_binary_feature_extractor_model_tflite_end");

const unsigned char *get_detector_model_tflite(size_t *len) {
    if (len) *len = _binary_detector_model_tflite_end - _binary_detector_model_tflite_start;
    return _binary_detector_model_tflite_start;
}
const unsigned char *get_feature_extractor_model_tflite(size_t *len) {
    if (len) *len = _binary_feature_extractor_model_tflite_end - _binary_feature_extractor_model_tflite_start;
    return _binary_feature_extractor_model_tflite_start;
}
