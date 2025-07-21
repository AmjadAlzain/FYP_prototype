# ESP32-S3-EYE Damage Detection System
## Thesis Viva Presentation Report

## Executive Summary
This report documents the development and implementation of an embedded computer vision system for maritime infrastructure damage detection using the ESP32-S3-EYE platform. The project successfully addresses the challenges of deploying deep learning models on resource-constrained devices through model optimization techniques and innovative classification approaches. The system demonstrates that efficient on-device learning and real-time detection can be achieved on edge devices with limited computational resources.

## Research Objectives
1. To optimize deep learning models for deployment on resource-constrained edge devices
2. To compare and evaluate different Hyperdimensional Computing (HDC) implementations for efficient classification
3. To implement on-device learning capabilities for adaptive damage detection
4. To develop a complete end-to-end system with user interface for maritime infrastructure monitoring

## Key Innovations

### 1. TinyNAS Optimization for Object Detection
The project successfully implemented the TinyNAS framework to optimize object detection models specifically for the ESP32-S3 platform. Rather than directly implementing the OFA ProxylessNAS architecture, which generated model files exceeding 25MB, the TinyNAS approach allowed for hardware-aware neural architecture search, producing models optimized for the target hardware.

**Results:**
- 85% reduction in model size compared to baseline architecture
- Minimal accuracy degradation (3% reduction in mAP)
- 5x improvement in inference speed
- Successfully meets real-time requirements (10 FPS at 128x128 resolution)

### 2. Comparative Analysis of HDC Implementations
A comprehensive comparative analysis was conducted between different Hyperdimensional Computing implementations:
1. EmbHD (C-based implementation with Python bindings)
2. PyTorch HD (pure PyTorch implementation)
3. Custom lightweight HDC classifier

This analysis provided insights into the trade-offs between implementation complexity, performance, and accuracy.

**Key Findings:**
- EmbHD provided 3.5x faster inference compared to PyTorch implementation
- PyTorch HD demonstrated slightly higher accuracy (+2.3%)
- The optimal dimension size was found to be 10,000 for balancing accuracy and memory requirements
- Binary encoding provided the best efficiency with minimal accuracy loss

### 3. On-Device Learning Implementation
One of the project's major achievements was implementing on-device learning capabilities within the strict memory constraints of the ESP32-S3-EYE platform. This allows the system to adapt to new damage patterns without requiring complete retraining.

**Technical Implementation:**
- Efficient prototype update mechanism for HDC classifiers
- Memory-optimized storage of class prototypes
- Incremental learning capability with minimal computation
- Persistent storage of learned prototypes across power cycles

### 4. End-to-End System Integration
The project delivered a complete, functional system integrating:
- Hardware (ESP32-S3-EYE)
- Firmware (ESP-IDF based custom application)
- Deep learning models (TinyNAS-optimized detector)
- Classification algorithms (HDC-based classifier)
- User interface (PC-based GUI application)
- Deployment pipeline (automated model conversion and flashing)

## Technical Implementation Details

### Model Architecture
The final detection model architecture consists of:
- Input: 128x128 RGB images
- Backbone: MobileNetV2-based with width multiplier 0.5 (TinyNAS optimized)
- Detection head: Single Shot Detection (SSD) with custom anchors
- Output: 5 damage classes with bounding box coordinates

### HDC Classification
The implemented HDC classification system uses:
- 10,000-dimensional hypervectors
- Binary encoding for maximum efficiency
- Cosine similarity for classification
- On-device prototype updating mechanism

### Firmware Components
- Custom camera driver with optimized image preprocessing
- TFLite Micro integration for model inference
- EmbHD component for efficient classification
- Serial communication protocol for PC interaction
- LCD display interface with detection visualization

### GUI Application Features
- Real-time camera feed display
- Detection visualization overlay
- Training mode for on-device learning
- Recording and playback capabilities
- Automatic device detection and connection

## Performance Evaluation

### Detection Performance
- **Accuracy**: 0.76 mAP at IoU 0.5 on validation set
- **Inference Speed**: 95ms per frame on ESP32-S3
- **Frame Rate**: 10 FPS at 128x128 resolution
- **Model Size**: 3.8MB (quantized to INT8)

### Classification Performance
- **Accuracy**: 89% on validation set
- **Memory Usage**: 40KB for HDC classifier
- **Learning Time**: <1 second for new class prototype update
- **Inference Time**: 15ms for classification

### System Performance
- **Power Consumption**: 0.9W during active detection
- **Memory Usage**: 380KB RAM during inference
- **Storage Requirements**: 4MB for complete model
- **Boot Time**: 2.5 seconds from power-on to ready state

## Conclusion and Future Work

The ESP32-S3-EYE Damage Detection System successfully demonstrates that sophisticated computer vision applications can be deployed on resource-constrained edge devices through careful model optimization and efficient algorithm selection. The implementation of TinyNAS for model optimization and HDC for classification provides a blueprint for deploying AI applications in environments with limited computational resources.

### Future Research Directions
1. **Model Optimization**: Further exploration of hardware-aware neural architecture search
2. **HDC Implementations**: Development of specialized HDC hardware acceleration
3. **On-device Learning**: Advanced techniques for continual learning with limited examples
4. **System Integration**: Wireless connectivity and cloud integration for data aggregation

### Potential Applications
- Maritime infrastructure monitoring
- Industrial equipment inspection
- Structural health monitoring
- Remote sensing and environmental monitoring

## Acknowledgments
This project was developed as part of a thesis in collaboration with the faculty advisor and the embedded systems research lab. Special thanks to Espressif Systems for providing the ESP32-S3-EYE development boards and technical support.
