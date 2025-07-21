# Enhanced ESP32-S3-EYE Container Detection System

## Overview

This enhanced GUI system provides **three inference modes** for container damage detection using your trained TinyNAS + HDC models:

1. **ESP32-S3-EYE Mode** - Hardware device communication
2. **Laptop Camera Mode** - Real-time webcam inference with 2-minute recording
3. **Video Upload Mode** - Batch processing of video files

## Architecture

```
Enhanced System Architecture
├── 🧠 Local Inference Engine (local_inference.py)
│   ├── TinyNAS Feature Extractor (256D features)
│   ├── HDC Classifier (2048D hypervectors)
│   └── Sliding Window Detection
├── 📷 Camera Manager (camera_manager.py)
│   ├── ESP32-S3-EYE Source
│   ├── Laptop Camera Source
│   └── Video File Source
├── 🖥️ Enhanced GUI (container_detection_gui_enhanced.py)
│   ├── Multi-mode Interface
│   ├── Real-time Detection Overlay
│   ├── Analytics Dashboard
│   └── Export Capabilities
└── 🔧 Utilities
    ├── Model Inspector
    └── Startup Scripts
```

## Features

### 🎯 Multi-Mode Operation
- **ESP32 Mode**: Direct hardware communication with your existing firmware
- **Laptop Camera Mode**: Real-time inference using laptop webcam
- **Video Upload Mode**: Process pre-recorded MP4 videos
- **Analytics Dashboard**: Statistics and detection history

### 🚀 Real-Time Performance
- **15 FPS processing** for supervisor demos
- **GPU acceleration** on RTX2060 (when available)
- **Multi-threaded processing** to prevent UI lag
- **Confidence scoring** with visual indicators

### 📊 Professional Interface
- **Modern PyQt6 design** with tabbed interface
- **Real-time detection overlays** with bounding boxes
- **2-minute recording mode** for demonstrations
- **Export capabilities** (video, CSV, PDF reports)

### 🔍 Detection Capabilities
- **5 damage classes**: axis, concave, dentado, perforation, no_damage
- **Sliding window detection** (64x64 patches)
- **Non-maximum suppression** for clean results
- **Confidence-based filtering** (adjustable threshold)

## Installation & Setup

### 1. Install Dependencies
```bash
# Navigate to Module 6 directory
cd 1_model_development/scripts/Modules/Module_6/

# Install enhanced requirements
pip install -r requirements_enhanced.txt
```

### 2. Verify Models
Your trained models should be in `1_model_development/models/`:
- `feature_extractor_fp32_best.pth` (TinyNAS backbone)
- `module3_hdc_model_embhd.npz` (HDC classifier)

### 3. Launch Enhanced GUI
```bash
# Windows
run_enhanced_gui.bat

# Or manually
python container_detection_gui_enhanced.py
```

## Usage Guide

### ESP32-S3-EYE Mode
1. Connect ESP32 device via USB
2. Select **ESP32-S3-EYE** tab
3. Choose COM port and click **Connect**
4. Use device control buttons for training/detection
5. View real-time inference results

### Laptop Camera Mode
1. Select **Laptop Camera** tab
2. Choose camera from dropdown
3. Click **Start Camera**
4. Use **Start 2-Minute Recording** for demos
5. Adjust FPS and overlay settings as needed

### Video Upload Mode
1. Select **Video Upload** tab
2. Click **Select Video File** (supports MP4, AVI, MOV)
3. Use playback controls to navigate
4. Click **Start Batch Analysis** for full processing
5. Export results using the export buttons

### Analytics Dashboard
1. Select **Analytics** tab
2. View real-time statistics and detection history
3. Monitor damage detection rates and confidence scores
4. Clear history as needed

## Technical Details

### Model Pipeline
```
Input Image (RGB) → Center Crop → Resize (64x64) → Normalize
      ↓
TinyNAS Feature Extractor → 256D Features
      ↓
HDC Projection Matrix → 2048D Hypervector
      ↓
HDC Classification → Damage Class + Confidence
```

### Detection Process
1. **Sliding Window**: 64x64 patches with 32px stride
2. **Feature Extraction**: TinyNAS backbone processes each patch
3. **HDC Classification**: Hyperdimensional computing for final prediction
4. **Post-processing**: NMS filtering and confidence thresholding
5. **Visualization**: Bounding boxes with class labels and confidence

### Performance Optimizations
- **Frame Queue**: Latest frame processing to avoid lag
- **GPU Acceleration**: CUDA support for RTX2060
- **Multi-threading**: Background inference worker
- **Efficient Preprocessing**: Optimized image operations

## File Structure

```
Module_6/
├── README_Enhanced.md              # This documentation
├── requirements_enhanced.txt       # Python dependencies
├── run_enhanced_gui.bat           # Windows startup script
├── local_inference.py             # Core inference engine
├── camera_manager.py              # Multi-source camera handling
├── container_detection_gui_enhanced.py  # Main GUI application
├── model_inspector.py             # Model analysis utility
└── Original Files/
    ├── container_detection_gui.py  # Original ESP32-only GUI
    ├── requirements.txt            # Original requirements
    └── run_gui.bat                # Original startup script
```

## Key Improvements

### vs. Original ESP32-Only GUI
✅ **Multi-mode operation** (ESP32 + Laptop + Video)  
✅ **Desktop inference** using same trained models  
✅ **2-minute recording** for supervisor demos  
✅ **Professional interface** with modern styling  
✅ **Analytics dashboard** with statistics  
✅ **Export capabilities** for reports  
✅ **Real-time performance** at 15 FPS  
✅ **GPU acceleration** support  

### Maintained Compatibility
✅ **ESP32 communication** protocol preserved  
✅ **Model architecture** identical to hardware  
✅ **Detection pipeline** matches firmware implementation  
✅ **Training workflows** remain unchanged  

## Troubleshooting

### Model Loading Issues
```bash
# Inspect your models
python model_inspector.py
```

### Camera Detection Problems
- Check camera permissions in Windows settings
- Try different camera indices (0, 1, 2...)
- Restart application if camera is in use

### Performance Issues
- Reduce FPS slider to 10-12 for slower systems
- Disable detection overlays for better performance
- Close other camera applications

### ESP32 Connection Issues
- Check USB cable and drivers
- Try different COM ports
- Ensure ESP32 firmware is properly flashed

## Model Requirements

Your system expects these trained models:
- **TinyNAS Feature Extractor**: PyTorch model (.pth)
- **HDC Classifier**: NumPy arrays (.npz) with projection matrix and class prototypes
- **Compatible preprocessing**: 64x64 RGB patches, ImageNet normalization

## Export Formats

### Video Export
- **Annotated MP4**: Original video with detection overlays
- **Frame extraction**: Individual frames with detections

### Report Export
- **CSV**: Timestamp, class, confidence, position data
- **PDF Summary**: Professional report with statistics
- **Screenshots**: Key frames with detection highlights

## Future Enhancements

🔄 **Planned Features**:
- Real-time model training from GUI
- Advanced filtering and search in analytics
- Custom detection thresholds per class
- Automated report generation
- Integration with cloud storage
- Multi-camera simultaneous processing

## Support

For technical issues:
1. Check the **Analytics** tab for system status
2. Review console output for error messages
3. Use **Model Inspector** to verify model files
4. Test with different input sources

## Performance Benchmarks

**Target Performance** (RTX2060 + 15 FPS):
- Feature extraction: ~10ms per patch
- HDC classification: ~1ms per patch
- Total processing: ~100ms per frame (640x480)
- Memory usage: ~2GB GPU, ~1GB RAM

**Supervisor Demo Mode**:
- 2-minute continuous recording
- Real-time detection overlays
- Automatic statistics generation
- Professional presentation quality

---

**Enhanced ESP32-S3-EYE Container Detection System v2.0**  
*Bridging embedded AI with desktop performance*
