# ESP32-S3-EYE Project Context Summary

This document provides a quick reference for the ESP32-S3-EYE Anomaly Detection System project, including key components, issues to address, and implementation strategies.

## Project Overview

The ESP32-S3-EYE Anomaly Detection System is a real-time edge AI system combining:
- MCUNet-based object detection
- Hyperdimensional Computing (HDC) classification
- On-device training capabilities

**Hardware Target**: ESP32-S3-EYE development board
- ESP32-S3 microcontroller (dual-core, up to 240MHz)
- 2MB PSRAM
- Built-in camera and 240x240 LCD display

## Project Structure

```
├── components/                 # ESP-IDF components
│   ├── embhd/                  # EmbHD hyperdimensional computing component
│   └── tflite_micro/           # TensorFlow Lite Micro component
├── main/                       # Main application code
├── Modules/                    # Training and preparation modules
│   ├── Module_1/               # Dataset preprocessing
│   ├── Module_2/               # MCUNet object detection model
│   ├── Module_3/               # HDC classifier implementation
│   ├── Module_4/               # Deployment and quantization
│   ├── Module_5/               # On-device training
│   └── Module_6/               # GUI and testing
├── Prototype/                  # PC-side tools
│   └── GUI/                    # Python GUI application
└── mcunet/                     # MCUNet TinyNAS implementation
```

## Key Issues to Address

### 1. MCUNet Model Optimization
- **Current Issue**: OFA Proxyless implementation generates large model files (25MB)
- **Requirements**: Localize TinyNAS and Tiny Transfer Learning frameworks
- **Goal**: Optimize for device constraints (fit within 2MB PSRAM)
- **Key Files**:
  - `Modules/Module_2/detection_model.py`
  - `Modules/Module_4/quantize_model.py`

### 2. HDC Classification Models
- **Current Issue**: Uncertainty about embHD vs direct PyTorch HD
- **Requirements**: Test different lightweight HDC classification models
- **Goal**: Find lightest and most suitable approach for resource constraints
- **Key Files**:
  - `Modules/Module_3/hdc_model.py`
  - `Modules/Module_3/train_hdc.py`
  - `Modules/Module_3/train_hdc_embhd.py`
  - `Modules/Module_3/embhd/` directory

### 3. GUI and Deployment
- **Current Issue**: Deployment code not working, GUI doesn't show camera output
- **Requirements**: Fix ESP32 connection detection and camera display
- **Goal**: Reliable deployment and user interface
- **Key Files**:
  - `Prototype/GUI/gui_app.py`
  - `deploy_to_esp32.py`
  - `main/app_camera.c`

## Implementation Strategy

### MCUNet Optimization
- Implement TinyNAS with custom constraints
- Apply model pruning to reduce size
- Use quantization-aware training
- Optimize for ESP32-S3 architecture

### HDC Model Selection
- Benchmark embHD vs PyTorch HD implementations
- Compare accuracy, memory usage, and inference speed
- Test different HD dimensions (10,000 vs 20,000)
- Select optimal approach based on results

### GUI and Deployment Fixes
- Fix frame transmission in SerialThread class
- Implement proper device connection detection
- Add port auto-detection
- Fix deployment script for reliable flashing

## Model Performance Requirements

### Training Speed Optimization
- Implement early stopping
- Use mixed precision training
- Apply learning rate scheduling
- Optimize batch sizes for faster convergence
- Implement data augmentation for better generalization

### Accuracy Requirements
- Maintain detection mAP of at least 65%
- Maintain classification accuracy of at least 90%
- Ensure consistent FPS of 15-20 on device

## Next Steps

1. Optimize MCUNet model architecture
2. Benchmark HDC implementations
3. Fix GUI camera display
4. Implement device detection
5. Create comprehensive documentation
6. Deploy and test the full system

## Documentation Deliverables

1. User Manual (usermanual.md)
2. Setup Guide (setup_guide.md)
3. Progress Report (progress_report.md)
4. Thesis Report (report.md)
