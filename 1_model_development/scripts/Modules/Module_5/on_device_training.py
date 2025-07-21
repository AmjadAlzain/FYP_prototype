# Module5/on_device_training.py
"""
on_device_training.py - Outlines the process for on-device training of the HDC model.
"""

# --- On-Device Training Workflow ---
# 1.  Capture a new image from the camera.
# 2.  Run the feature extractor model to get the feature vector for the new image.
# 3.  The user provides the correct label for the new image through a GUI or serial command.
# 4.  The feature vector and the correct label are passed to the on-device training function.
# 5.  The on-device training function updates the HDC model's prototypes.
# 6.  The updated prototypes are saved to flash memory to be used for future inferences.

# --- C Implementation for On-Device Training ---
# The following C code would be added to the `embhd` component to support on-device training.

"""
// In embhd.h
/**
 * @brief Updates a class prototype with a new feature vector.
 *
 * @param class_id The class to update.
 * @param feature_vector The feature vector of the new sample.
 * @param learning_rate The learning rate for the update.
 */
void embhd_update_prototype(int class_id, int8_t* feature_vector, float learning_rate);

// In embhd.c
#include "embhd_prototypes.h" // This would now need to be non-const

void embhd_update_prototype(int class_id, int8_t* feature_vector, float learning_rate) {
    if (class_id < 0 || class_id >= EMBHD_NUM_CLASSES) {
        return; // Invalid class ID
    }

    // This is a simplified example. A real implementation would need to
    // handle the learning rate and update the prototype vector in flash.
    for (int i = 0; i < EMBHD_VECTOR_DIM; i++) {
        // A simple update rule: move the prototype towards the new vector
        embhd_prototypes[class_id][i] = (1.0 - learning_rate) * embhd_prototypes[class_id][i] + learning_rate * feature_vector[i];
    }
}
"""

# --- Main Application Logic ---
# The main application loop in `main.cpp` would be modified to include a trigger for on-device training.
# This could be a button press or a command received over serial.

"""
// In main.cpp
void check_for_training_trigger() {
    // Check for a button press or serial command
    if (training_triggered) {
        // 1. Get the last captured feature vector
        int8_t* last_feature_vector = feature_extractor_interpreter->output(0)->data.int8;

        // 2. Get the correct label from the user
        int correct_label = get_label_from_user();

        // 3. Update the HDC model
        embhd_update_prototype(correct_label, last_feature_vector, 0.1);

        // 4. Save the updated prototypes to flash
        save_prototypes_to_flash();
    }
}
"""
