/**
 * embhd_esp32.h - ESP32-S3-EYE specific adaptations for EmbHD
 * 
 * This header provides ESP32-specific implementations and optimizations
 * for the EmbHD library, targeting ESP32-S3-EYE specifically.
 */

#ifndef EMBHD_ESP32_H
#define EMBHD_ESP32_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "esp_system.h"
#include "esp_log.h"
#include "embhd.h"

// Configuration for ESP32-S3-EYE
#define EMBHD_ESP32_USE_PSRAM 1  // Use PSRAM for large matrices if available
#define EMBHD_ESP32_LOG_TAG "EmbHD"

// Forward declarations
typedef struct embhd_esp32_context_t embhd_esp32_context_t;

/**
 * ESP32-specific context for EmbHD operations
 */
struct embhd_esp32_context_t {
    embhd_model_t model;         // The EmbHD model
    bool initialized;            // Initialization status
    bool using_psram;            // Whether PSRAM is being used
    int8_t last_prediction;      // Last prediction result
    float* last_scores;          // Last classification scores
};

/**
 * Initialize EmbHD model on ESP32
 * 
 * @param ctx Context to initialize
 * @param in_features Input feature dimensions
 * @param num_classes Number of classes
 * @param hd_dim HD vector dimensions (default = 10000 for ESP32)
 * @param vtype Vector type (default = EMBHD_BIPOLAR)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t embhd_esp32_init(embhd_esp32_context_t* ctx, 
                          uint32_t in_features, 
                          uint32_t num_classes,
                          uint32_t hd_dim,
                          embhd_vtype_t vtype);

/**
 * Free resources used by the EmbHD model
 * 
 * @param ctx EmbHD ESP32 context
 */
void embhd_esp32_deinit(embhd_esp32_context_t* ctx);

/**
 * Initialize EmbHD model from pre-trained headers
 * 
 * The headers should define:
 * - EMBHD_IN_FEATURES: Input dimensions
 * - EMBHD_OUT_FEATURES: HD dimensions
 * - EMBHD_NUM_CLASSES: Number of classes
 * - proj_matrix: Projection matrix
 * - prototypes: Class prototypes
 * 
 * @param ctx Context to initialize
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t embhd_esp32_init_from_headers(embhd_esp32_context_t* ctx);

/**
 * Predict class for input features
 * 
 * @param ctx EmbHD ESP32 context
 * @param features Input feature vector
 * @param scores Optional array to store similarity scores (can be NULL)
 * @return Predicted class ID (negative value on error)
 */
int8_t embhd_esp32_predict(embhd_esp32_context_t* ctx, 
                         const float* features, 
                         float* scores);

/**
 * Train model on a single example (on-device learning)
 * 
 * @param ctx EmbHD ESP32 context
 * @param features Input feature vector
 * @param class_id Class ID
 * @param learning_rate Learning rate for update (0.0-1.0)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t embhd_esp32_train_sample(embhd_esp32_context_t* ctx,
                                 const float* features,
                                 uint32_t class_id,
                                 float learning_rate);

/**
 * Get memory usage statistics for EmbHD model
 * 
 * @param ctx EmbHD ESP32 context
 * @param ram_usage Pointer to store RAM usage in bytes
 * @param psram_usage Pointer to store PSRAM usage in bytes
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t embhd_esp32_get_memory_usage(embhd_esp32_context_t* ctx,
                                     uint32_t* ram_usage,
                                     uint32_t* psram_usage);

#endif /* EMBHD_ESP32_H */
