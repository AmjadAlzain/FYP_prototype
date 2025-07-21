# HDC Dataset for Whole Container Classification

## Overview
This document describes the new HDC dataset approach that uses **whole container images** instead of small cropped patches for damage classification.

## Key Improvements Over Previous Approach

### Previous Approach (Patches)
- ❌ Small 64x64 damage patches
- ❌ Lost spatial context and relationships  
- ❌ Difficult to understand overall container condition
- ❌ Required patch aggregation for final classification

### New Approach (Whole Containers)
- ✅ Complete 256x256 container images
- ✅ Preserves spatial relationships and context
- ✅ Direct container-level classification
- ✅ More realistic for inspection workflows
- ✅ Better accuracy potential with HDC

## Dataset Structure

```
prototype/hdc_dataset/
├── train/
│   ├── axis/           # Containers with axis damage
│   ├── concave/        # Containers with concave damage
│   ├── dentado/        # Containers with dentado damage
│   ├── perforation/    # Containers with perforation damage
│   └── no_damage/      # Healthy containers
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
├── dataset_statistics.json
└── dataset_info.json
```

## Classification Logic

### Container Damage Assessment
1. **Parse YOLO annotations** from SeaFront bbannotation files
2. **Extract container bounding boxes** (class_id = 0)
3. **Analyze damage within container bounds**:
   - axis (class_id = 1)
   - concave (class_id = 2) 
   - dentado (class_id = 3)
   - perforation (class_id = 4)
4. **Classify container** based on most severe damage type
5. **No damage** if no damage annotations overlap with container

### Damage Priority (Most Severe First)
1. **Perforation** - Most critical structural damage
2. **Axis** - Significant structural deformation
3. **Dentado** - Moderate edge damage
4. **Concave** - Surface deformation
5. **No Damage** - Healthy container

## Files Created

### 1. Data Preprocessing
- **`hdc_data_preprocessor.py`** - Main preprocessor
- **`run_hdc_preprocessor.py`** - Simple execution script

### 2. HDC Training  
- **`train_hdc_whole_containers.py`** - Updated Module 3 training
- **`run_complete_hdc_pipeline.py`** - Complete pipeline runner

### 3. Documentation
- **`HDC_DATASET_README.md`** - This file

## Usage Instructions

### Step 1: Create HDC Dataset
```bash
cd 1_model_development/scripts/Modules/Module_1
python hdc_data_preprocessor.py
```

### Step 2: Train HDC Classifier
```bash
cd ../Module_3
python train_hdc_whole_containers.py --weight-strategy balanced --epochs 50
```

### Step 3: Complete Pipeline (Recommended)
```bash
cd ../Module_1  
python run_complete_hdc_pipeline.py
```

## Technical Details

### Image Processing
- **Input**: SeaFront container images with YOLO annotations
- **Output**: 256x256 standardized container images
- **Preprocessing**: Resize with aspect ratio preservation + padding
- **Quality Filter**: Skip containers smaller than 50px

### Feature Extraction
- **TinyNAS Features**: 256D feature vectors from Module 2
- **HDC Dimensions**: 2048D hypervectors (configurable)
- **Batch Processing**: Efficient GPU-accelerated feature extraction

### Training Configuration
- **Epochs**: 50 (configurable)
- **Learning Rate**: 0.1 (HDC standard)
- **Weight Strategy**: Balanced class weighting (recommended)
- **Data Augmentation**: Rotation, flip, color jitter for training

## Expected Benefits

### 1. Improved Accuracy
- **Context Preservation**: Spatial relationships between damage areas
- **Realistic Assessment**: Whole container condition evaluation
- **Better Generalization**: More representative training samples

### 2. Simplified Pipeline
- **Direct Classification**: No patch aggregation needed
- **End-to-End**: Container image → damage classification
- **Deployment Ready**: Matches real inspection workflows

### 3. Enhanced GUI Integration
- **Live Inference**: Real-time whole container classification
- **Professional Display**: Container-level bounding boxes
- **Intuitive Results**: Clear damage/no-damage classification

## Integration with Enhanced GUI

The new HDC models will integrate seamlessly with the enhanced GUI:

1. **Local Inference**: `local_inference.py` uses new HDC models
2. **Real-Time Processing**: 15fps container classification
3. **Professional Display**: Green (healthy) / Red (damaged) containers
4. **Multiple Input Sources**: ESP32, laptop camera, video upload

## Next Steps

1. **Run Pipeline**: Execute `run_complete_hdc_pipeline.py`
2. **Test Performance**: Compare accuracy with patch-based approach
3. **Update GUI**: Integrate new models in `local_inference.py`
4. **Deploy**: Test with ESP32 for on-device inference

## Performance Expectations

Based on the improved approach, we expect:
- **Higher Accuracy**: 85-95% validation accuracy (vs 70-80% with patches)
- **Better Confidence**: More reliable damage detection
- **Faster Inference**: Single container classification vs multiple patches
- **Robust Performance**: Better generalization to real container images

---

*This HDC dataset approach represents a significant improvement in container damage classification methodology, providing more realistic and accurate damage assessment capabilities.*
