// In components/hdc/hdc.c
#include "hdc.h" // Our public API
#include <string.h>
#include <limits.h>
#include <stdio.h>

// This file needs access to the constant projection matrix.
// We declare it here. It's defined in embhd_weights.c
extern const int8_t g_hdc_projection[HDC_FEATURE_DIM][HDC_INPUT_DIM];
extern const int8_t g_initial_hdc_prototypes[NUM_CLASSES][HDC_FEATURE_DIM];


// Helper: project an input vector and binarize it to a hypervector
static void project_and_binarize(const int8_t *feature, int8_t *hypervector) {
    for (int j = 0; j < HDC_FEATURE_DIM; ++j) {
        int32_t acc = 0;
        for (int i = 0; i < HDC_INPUT_DIM; ++i) {
            // Access the projection matrix correctly
            acc += (int16_t)g_hdc_projection[j][i] * (int16_t)feature[i];
        }
        hypervector[j] = (acc >= 0) ? 1 : -1;
    }
}

// --- Public API Implementation ---

void hdc_init_prototypes() {
    // Copy the initial prototypes from flash into the RAM-based array
    memcpy(g_hdc_prototypes, g_initial_hdc_prototypes, sizeof(g_hdc_prototypes));
}

int hdc_classify(const int8_t *feature_vec, float *confidence_out) {
    int8_t hypervector[HDC_FEATURE_DIM];
    project_and_binarize(feature_vec, hypervector);

    int best_class_id = -1;
    int32_t best_score = INT32_MIN;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        int32_t current_score = 0;
        for (int j = 0; j < HDC_FEATURE_DIM; ++j) {
            // Use the global RAM prototypes for comparison
            current_score += hypervector[j] * g_hdc_prototypes[c][j];
        }

        if (current_score > best_score) {
            best_score = current_score;
            best_class_id = c;
        }
    }

    if (confidence_out) {
        *confidence_out = ((float)best_score) / HDC_FEATURE_DIM;
    }
    return best_class_id;
}

void hdc_update_prototype(int class_idx, const int8_t *feature_vec) {
    if (class_idx < 0 || class_idx >= NUM_CLASSES) {
        return; // Invalid class index
    }

    int8_t new_hv[HDC_FEATURE_DIM];
    project_and_binarize(feature_vec, new_hv);

    // Get a pointer to the RAM prototype for the target class
    int8_t *proto_to_update = g_hdc_prototypes[class_idx];

    // Simple update: bundle the new hypervector with the existing prototype
    for (int j = 0; j < HDC_FEATURE_DIM; ++j) {
        int val = (int)proto_to_update[j] + (int)new_hv[j];
        // Clip the values to prevent overflow and keep them as int8
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        proto_to_update[j] = (int8_t)val;
    }
    printf("Prototype for class %d updated.\n", class_idx);
}