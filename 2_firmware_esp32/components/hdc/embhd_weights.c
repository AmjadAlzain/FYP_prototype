// In components/hdc/embhd_weights.c
#include <stdint.h>
#include "hdc_config.h"
#include "embhd_projection.h"
#include "embhd_prototypes.h"

// Temporarily disable the "-Wmissing-braces" warning for this auto-generated data.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"

/**
 * @brief Defines the constant projection matrix, stored in flash.
 */
const int8_t g_hdc_projection[HDC_FEATURE_DIM][HDC_INPUT_DIM] =
    { EMBHD_PROJ_INITIALIZER };

/**
 * @brief Defines the initial class prototypes, stored in flash.
 */
const int8_t g_initial_hdc_prototypes[NUM_CLASSES][HDC_FEATURE_DIM] =
    { EMBHD_PROTO_INITIALIZER };

// Restore the compiler's default warnings
#pragma GCC diagnostic pop


/**
 * @brief RAM-based prototypes for online learning.
 */
int8_t g_hdc_prototypes[NUM_CLASSES][HDC_FEATURE_DIM];