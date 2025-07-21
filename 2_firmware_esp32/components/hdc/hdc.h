// In components/hdc/hdc.h
#pragma once

#include <stdint.h>
#include "esp_err.h"
#include "hdc_config.h" // Include the centralized dimensions

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the HDC prototypes from flash to RAM.
 * Must be called once at startup before any other HDC functions.
 */
void hdc_init_prototypes(void);

/**
 * @brief Classifies an input feature vector using the HDC model.
 *
 * @param feature_vec       Pointer to the input feature vector (size: HDC_INPUT_DIM).
 * @param confidence_out    Pointer to a float to store the confidence score (can be NULL).
 * @return int              The predicted class ID.
 */
int hdc_classify(const int8_t *feature_vec, float *confidence_out);

/**
 * @brief Updates a class prototype with a new feature vector for online learning.
 *
 * @param class_idx         The class ID of the prototype to update.
 * @param feature_vector    The new feature vector to learn from.
 */
void hdc_update_prototype(int class_idx, const int8_t *feature_vector);


/**
 * @brief Global array for class prototypes, stored in RAM.
 *
 * Declared here as 'extern' so other components can access it for updates,
 * but it is DEFINED in embhd_weights.c.
 */
extern int8_t g_hdc_prototypes[NUM_CLASSES][HDC_FEATURE_DIM];


#ifdef __cplusplus
}
#endif