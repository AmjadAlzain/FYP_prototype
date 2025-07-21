# ESP32-S3-EYE Container Anomaly Detection System
## Complete Deployment Guide

## 🎯 Overview
This guide provides step-by-step instructions for deploying the complete container anomaly detection system on the ESP32-S3-EYE board, including firmware flashing, model integration, and GUI setup.

## 📋 Prerequisites

### Hardware Requirements
- **ESP32-S3-EYE development board** (with built-in camera and LCD)
- **USB-C cable** for programming and power
- **MicroSD card** (optional, for additional storage)
- **Windows PC** with available COM ports

### Software Requirements
- **ESP-IDF v5.4.1** (dirty build supported)
- **Python 3.8+** with pip
- **Git** for repository management
- **VSCode** with ESP-IDF extension (recommended)

## 🛠️ Installation Steps

### Step 1: Environment Setup

1. **Install ESP-IDF v5.4.1**
   ```bash
   # Download ESP-IDF
   git clone --recursive https://github.com/espressif/esp-idf.git
   cd esp-idf
   git checkout v5.4.1
   git submodule update --init --recursive
   
   # Install (Windows)
   install.bat esp32s3
   
   # Set up environment
   export.bat
   ```

2. **Install Python Dependencies**
   ```bash
   pip install pyqt6 opencv-python numpy pyserial
   ```

### Step 2: Project Setup

1. **Navigate to Firmware Directory**
   ```bash
   cd f:/FYP/Prototype/2_firmware_esp32
   ```

2. **Clean Previous Builds**
   ```bash
   idf.py fullclean
   ```

3. **Configure Project**
   ```bash
   idf.py set-target esp32s3
   idf.py menuconfig
   ```

### Step 3: Build and Flash

1. **Build the Project**
   ```bash
   idf.py build
   ```

2. **Flash to ESP32-S3-EYE**
   ```bash
   idf.py -p COM3 flash monitor
   ```
   *(Replace COM3 with your actual port)*

## 🧠 Model Integration

### TFLite Models
The firmware automatically loads these models from components/model_data/:
- **`detector_model.tflite`** - Container detection (105KB arena)
- **`feature_extractor_model.tflite`** - Feature extraction (135KB arena)

### HDC Classification
The HDC model components are loaded from components/embhd/:
- **`embhd_projection.h`** - 256→4096 dimensional projection matrix
- **`embhd_prototypes.h`** - Class prototypes for 5 damage types
- **`embhd.c/.h`** - Core HDC implementation with training

### Memory Allocation
```cpp
// PSRAM allocation for models
Detector Arena:         105KB (SPIRAM)
Feature Extractor:      135KB (SPIRAM) 
HDC Projections:        4KB (Flash)
HDC Prototypes:         1KB (Flash)
Total Model Memory:     ~245KB
```

## 🎮 System Operation

### Button Controls (ESP32-S3-EYE)
- **GPIO 0 (Boot)**: Capture frame / Cancel operation
- **GPIO 47**: Load from storage / Navigate up
- **GPIO 48**: Classify as "No Damage" / Navigate down  
- **GPIO 45**: Classify as "Damage" / Confirm selection

### Operation Modes

#### 1. Live Detection Mode (Default)
- Real-time camera feed with overlay detection
- Container detection with green/red bounding boxes
- Individual damage patches with color-coded types:
  - 🟡 **Yellow**: Dentado damage
  - 🟣 **Purple**: Concave damage  
  - 🟪 **Pink**: Axis damage
  - 🔵 **Blue**: Perforation damage
  - 🟢 **Green**: No damage

#### 2. Training Mode
- Press **Button 1** to freeze current frame
- Press **Button 3** for "No Damage" training
- Press **Button 4** to select specific damage type
- HDC model updates on-device using `embhd_train_on_sample()`

#### 3. Storage Review Mode  
- Press **Button 2** to load unclassified images
- Navigate and classify stored images
- Automatic file management and cleanup

## 🖥️ Desktop GUI Application

### Starting the GUI
```bash
cd 1_model_development/scripts/Modules/Module_6
python esp32_gui.py
```

### GUI Features
- **Real-time Video Feed**: Live camera stream with detection overlays
- **Serial Communication**: Auto-detection and connection to ESP32
- **Training Controls**: Remote training commands via serial
- **Performance Monitoring**: FPS, memory usage, detection counts
- **Event Logging**: Timestamped system events and errors

### Connection Setup
1. Launch GUI application
2. Select correct COM port from dropdown
3. Click "Connect" button
4. Monitor connection status in status panel

## 🔧 Configuration Files

### Memory Configuration (sdkconfig.defaults)
```ini
# Flash and PSRAM
CONFIG_ESPTOOLPY_FLASHSIZE_8MB=y
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_USE_CAPS_ALLOC=y

# Camera and Display  
CONFIG_CAMERA_CORE0=y
CONFIG_LV_COLOR_DEPTH_16=y

# TensorFlow Lite
CONFIG_TFLITE_MICRO_OPTIMIZATIONS=y
```

### Partition Table (partitions.csv)
```csv
# Name,   Type, SubType, Offset,  Size,     Flags
nvs,      data, nvs,     0x9000,  0x6000,
phy_init, data, phy,     0xf000,  0x1000,
factory,  app,  factory, 0x10000, 0x500000,
storage,  data, spiffs,  0x510000,0x2F0000,
```

### Dependencies (idf_component.yml)
```yaml
dependencies:
  espressif/esp32_s3_eye: "*"
  espressif/esp-tflite-micro: "*"  
  espressif/esp32-camera: "*"
  espressif/button: "*"
  espressif/esp_lvgl_port: "*"
  espressif/esp_jpeg: "*"
```

## 📊 Performance Specifications

### Detection Performance
- **Container Detection**: ~150ms per frame
- **Damage Classification**: ~50ms per patch
- **Overall Latency**: <500ms end-to-end
- **Classification Accuracy**: 98.24% (test set)

### Resource Usage
- **Flash Memory**: ~2.5MB (models + firmware)
- **PSRAM Usage**: ~245KB (model arenas)
- **SRAM Usage**: ~180KB (runtime data)
- **Power Consumption**: ~400mA @ 3.3V

### Supported Damage Classes
1. **axis** - Structural axis damage
2. **concave** - Dented/concave surfaces  
3. **dentado** - Jagged/serrated damage
4. **perforation** - Holes and punctures
5. **no_damage** - Undamaged containers

## 🚨 Troubleshooting

### Common Issues

#### Build Errors
```bash
# Clear cache and rebuild
idf.py fullclean
idf.py build
```

#### Memory Issues
```bash
# Check PSRAM configuration
idf.py menuconfig
# Component config → ESP32S3-Specific → Support for external SPI RAM
```

#### Camera Not Working
```bash
# Verify camera configuration
idf.py menuconfig  
# Component config → Camera configuration → Camera pinout
```

#### Serial Connection Issues
```bash
# Check port and drivers
Device Manager → Ports (COM & LPT)
# Install CP210x drivers if needed
```

### Debug Commands
```bash
# Monitor serial output
idf.py -p COM3 monitor

# Check partition table
idf.py partition-table

# Analyze binary size
idf.py size

# Check memory usage
idf.py size-components
```

## 🧪 Testing and Validation

### Firmware Testing
1. **Boot Test**: Verify startup logs and model loading
2. **Camera Test**: Check live video feed display
3. **Detection Test**: Test with known container images
4. **Training Test**: Verify on-device learning functionality
5. **Storage Test**: Test SD card and SPIFFS operations

### End-to-End Pipeline Test
```bash
cd 1_model_development/scripts
python test_end_to_end.py
```

### Expected Outputs
- Detection logs in serial monitor
- Color-coded bounding boxes on display
- Training confirmations in logs
- GUI status updates

## 📈 Performance Optimization

### For Better Detection
- Ensure good lighting conditions
- Position containers clearly in frame
- Use 240x240 resolution for optimal balance
- Maintain stable camera positioning

### For Training Accuracy  
- Collect diverse training samples
- Balance damage type distributions
- Use clear, unambiguous examples
- Train in similar lighting conditions

### Memory Optimization
- Monitor PSRAM usage during operation
- Clear detection buffers regularly
- Optimize image processing pipeline
- Use appropriate model quantization

## 🔮 Future Enhancements

### Short-term Improvements
- **Multi-damage Detection**: Detect multiple damage types per container
- **Confidence Thresholding**: Adjustable detection sensitivity
- **Performance Metrics**: Real-time accuracy monitoring
- **Data Logging**: Comprehensive detection history

### Long-term Vision
- **Wireless Communication**: WiFi/Bluetooth connectivity
- **Cloud Integration**: Remote model updates
- **Fleet Management**: Multi-device coordination
- **Advanced Analytics**: Trend analysis and reporting

## 📞 Support and Maintenance

### Regular Maintenance
- **Monthly**: Update ESP-IDF components
- **Quarterly**: Retrain models with new data
- **As needed**: Firmware updates for bug fixes

### Getting Help
- **Documentation**: Check project README and code comments
- **Logs**: Monitor serial output for error details
- **Community**: ESP32 forums and documentation
- **Repository**: GitHub issues and discussions

---

**🎉 Congratulations!** Your ESP32-S3-EYE Container Anomaly Detection System is now ready for deployment. The system provides real-time, accurate container damage detection with on-device learning capabilities.
