# EmbHD - Embedded Hyperdimensional Computing Library

EmbHD is a lightweight hyperdimensional computing (HDC) library designed specifically for embedded systems like the ESP32-S3-EYE. It enables efficient machine learning on resource-constrained devices.

## Overview

This library provides:

1. **C Implementation**: Core HDC functions optimized for embedded systems
2. **Python Wrapper**: For training and testing on desktop before deployment
3. **ESP32 Adaptation**: ESP32-specific optimizations and memory management
4. **Model Export**: Tools to export trained models to C headers for ESP32

## Components

- `embhd.h` / `embhd.c`: Core C implementation of HDC algorithms
- `embhd_py.py`: Python wrapper for training and model conversion
- `esp32/`: ESP32-specific adaptations and optimizations
- `headers/`: Auto-generated C headers from trained models

## Training a Model

To train an HDC model using EmbHD:

1. Extract features from your dataset
2. Run the training script:

```bash
python Modules/Module_3/train_hdc_embhd.py
```

This will:
- Load pre-extracted features
- Train an HDC model using EmbHD
- Save the model to `module3_hdc_model_embhd.pth`
- Export C headers for ESP32 deployment

## Integration with ESP32

To use EmbHD in your ESP32 project:

1. Copy the following files to your ESP-IDF project:
   - `embhd.h`
   - `embhd.c`
   - `esp32/embhd_esp32.h`
   - `esp32/embhd_esp32.c`
   - Generated headers from `headers/`

2. Include the component in your ESP-IDF project
3. Use the API to perform classification:

```c
#include "embhd_esp32.h"

// Initialize the model
embhd_esp32_context_t model_ctx;
embhd_esp32_init_from_headers(&model_ctx);

// Extract features from input (e.g., using CNN)
float features[EMBHD_IN_FEATURES];
extract_features(input_data, features);

// Classify
int8_t prediction = embhd_esp32_predict(&model_ctx, features, NULL);
printf("Predicted class: %d\n", prediction);
```

## On-Device Training

EmbHD supports on-device training for continual learning:

```c
// Train on a new example
float features[EMBHD_IN_FEATURES];
uint32_t class_id = 5;
embhd_esp32_train_sample(&model_ctx, features, class_id, 0.1f);
```

## Memory Usage

The library is optimized for minimal RAM usage:

- Uses int8_t for bipolar vectors (-1/+1)
- Supports bit-packing for binary vectors
- Automatically uses PSRAM when available on ESP32-S3

## Parameters

Key parameters that affect model performance:

- `HDC_DIMENSIONS`: Higher values give better accuracy but use more memory
- `EMBHD_VTYPE`: Vector type (BINARY, BIPOLAR, or FLOAT)
- `TRAIN_EPOCHS`: More epochs improve accuracy but take longer to train

## Integration with Module 5 (On-Device Training)

The EmbHD library is designed to work with the on-device training implementation in Module 5. Simply use the `embhd_esp32_train_sample` function to update the model with new examples.
