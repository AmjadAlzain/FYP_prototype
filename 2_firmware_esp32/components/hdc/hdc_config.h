// In components/hdc/hdc_config.h
#pragma once

// -- MODEL CONFIGURATION --
// This is the single source of truth for the model's dimensions.
// Your Python export script shows these values are correct for your trained model.
#define HDC_INPUT_DIM       256   // The dimension of the feature vector from the TFLite model.
#define HDC_FEATURE_DIM     2048  // The dimension of the hypervector after projection.
#define NUM_CLASSES         5     // The number of classes in your dataset.