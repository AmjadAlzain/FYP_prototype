// components/model_data/model_data.h
#pragma once

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

const unsigned char *get_detector_model_tflite(size_t *len);
const unsigned char *get_feature_extractor_model_tflite(size_t *len);

#ifdef __cplusplus
}
#endif
