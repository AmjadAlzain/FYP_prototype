# ESP32-S3-EYE Damage Detection System
## Step-by-Step Setup Guide

This guide provides detailed instructions for setting up the ESP32-S3-EYE Damage Detection System from scratch, including hardware setup, software installation, model training, and deployment.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Hardware Setup](#hardware-setup)
3. [Development Environment Setup](#development-environment-setup)
4. [Model Training and Optimization](#model-training-and-optimization)
5. [ESP32-S3-EYE Firmware Setup](#esp32-s3-eye-firmware-setup)
6. [Deployment](#deployment)
7. [GUI Application Setup](#gui-application-setup)
8. [Verification and Testing](#verification-and-testing)

## Prerequisites

### Required Hardware
- ESP32-S3-EYE development board
- USB-C cable for programming and power
- Computer with USB port (Windows, macOS, or Linux)
- (Optional) USB hub if multiple devices need to be connected

### Required Software
- Python 3.7 or later
- ESP-IDF (Espressif IoT Development Framework) v4.4 or later
- Git for version control
- VSCode or another IDE for development

## Hardware Setup

1. **Unbox the ESP32-S3-EYE Development Board**
   - Carefully remove the board from its packaging
   - Inspect for any physical damage

2. **Connect ESP32-S3-EYE to Computer**
   - Use the USB-C cable to connect the ESP32-S3-EYE to your computer
   - The board should power on, showing the boot screen on its LCD

3. **Verify Initial Connection**
   - On Windows, check Device Manager to ensure the board appears as a COM port
   - On macOS, check if it appears in the `/dev` directory as a `cu.` device
   - On Linux, check if it appears in the `/dev` directory as a `ttyUSB` device

## Development Environment Setup

### Installing Python Dependencies

1. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install Required Python Packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   If the requirements.txt file is not available, install these packages:
   ```bash
   pip install torch torchvision numpy opencv-python pyserial pillow matplotlib tqdm
   ```

### Installing ESP-IDF

#### Windows
1. Download and run the ESP-IDF Windows Installer from [Espressif website](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/windows-setup.html)
2. Follow the installer instructions to set up the ESP-IDF environment
3. Verify installation by opening the ESP-IDF Command Prompt and running:
   ```
   idf.py --version
   ```

#### macOS and Linux
1. Install prerequisites:
   ```bash
   # macOS
   brew install cmake ninja dfu-util
   
   # Linux (Ubuntu)
   sudo apt-get install git wget flex bison gperf python3 python3-pip python3-setuptools cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
   ```

2. Clone the ESP-IDF repository:
   ```bash
   mkdir -p ~/esp
   cd ~/esp
   git clone -b v4.4.3 --recursive https://github.com/espressif/esp-idf.git
   ```

3. Run the installation script:
   ```bash
   cd ~/esp/esp-idf
   ./install.sh esp32s3
   ```

4. Set up environment variables:
   ```bash
   . $HOME/esp/esp-idf/export.sh
   ```

## Model Training and Optimization

### Dataset Preparation

1. **Organize Your Dataset**
   - Run the dataset organization script:
     ```bash
     python Modules/Module_1/organize_labels.py
     ```
   - Visualize the dataset to ensure correctness:
     ```bash
     python Modules/Module_1/visualize_dataset.py
     ```

### Detection Model Training

1. **Compute Anchor Boxes**
   ```bash
   python Modules/Module_2/compute_anchors.py
   ```

2. **Train TinyNAS Detection Model (Recommended)**
   ```bash
   python Modules/Module_2/train_tinynas.py --width_mult 0.5 --batch_size 16 --epochs 50
   ```

   This will create a model file: `module2_tinynas_detection_model_best.pth`

3. **Alternatively, Train Standard Detection Model**
   ```bash
   python Modules/Module_2/train_detection.py
   ```

4. **Evaluate Detection Model**
   ```bash
   python Modules/Module_2/evaluate.py --model_path module2_tinynas_detection_model_best.pth
   ```

### HDC Classifier Training

1. **Benchmark Different HDC Models**
   ```bash
   python Modules/Module_3/benchmark_hdc.py --test-all --generate-plots
   ```

2. **Train HDC Model (Choose Based on Benchmark Results)**
   - For EmbHD implementation:
     ```bash
     python Modules/Module_3/train_hdc_embhd.py
     ```
   - For PyTorch HD implementation:
     ```bash
     python Modules/Module_3/train_hdc.py --use-pytorch-hd --hdc-dim 10000
     ```

### Model Optimization

1. **Quantize the Model**
   ```bash
   python Modules/Module_4/quantize_model.py --model_path module2_tinynas_detection_model_best.pth --output_path module4_backbone_quantized.pth
   ```

## ESP32-S3-EYE Firmware Setup

1. **Clone the Repository** (If you haven't already)
   ```bash
   git clone https://github.com/yourusername/esp32-s3-eye-damage-detection.git
   cd esp32-s3-eye-damage-detection
   ```

2. **Prepare Components Directory**
   - Ensure the `components` directory exists with the following subdirectories:
     - `embhd` - For Hyperdimensional Computing
     - `tflite_micro` - For TensorFlow Lite Micro

3. **Set Up the Main Application**
   - Review the main application code in the `main` directory
   - Make sure app_camera.c, app_lcd.c, and main.c are correctly configured

## Deployment

1. **Prepare for Deployment**
   ```bash
   python prepare_for_deployment.py
   ```
   This script prepares all necessary files for ESP32-S3-EYE deployment.

2. **Deploy to ESP32-S3-EYE**
   ```bash
   python deploy_to_esp32.py
   ```
   
   This script:
   - Converts PyTorch models to TFLite format
   - Converts HDC model to EmbHD format
   - Generates C headers with model data
   - Builds and flashes firmware to ESP32-S3-EYE

3. **Alternative Deployment (Windows Batch File)**
   ```bash
   deploy_esp32.bat
   ```

## GUI Application Setup

1. **Navigate to the GUI Directory**
   ```bash
   cd Prototype/GUI
   ```

2. **Install GUI Dependencies**
   ```bash
   pip install pyserial opencv-python pillow numpy matplotlib
   ```

3. **Launch the GUI Application**
   ```bash
   python gui_app.py
   ```

4. **Connect to ESP32-S3-EYE**
   - In the GUI, enter "AUTO" in the port field for automatic detection
   - Click "Connect"
   - The camera feed should appear in the display area

## Verification and Testing

1. **Run the Onboard Test Suite**
   - In the GUI, click "Run Test Suite"
   - Check the log area for test results

2. **Verify Object Detection**
   - Point the camera at objects that should be detected
   - Verify that bounding boxes appear around detected objects

3. **Test On-device Training**
   - Select a class from the dropdown menu
   - Click "Start Training"
   - Show multiple examples of the class to the camera
   - Click "Stop Training"
   - Test if the system now recognizes the trained class

4. **Record a Test Session**
   - Click "Start Recording"
   - Perform various detection operations
   - Click "Stop Recording"
   - Verify that the recording was saved correctly

## Troubleshooting

- If the ESP32-S3-EYE is not detected, try different USB ports or cables
- If the firmware fails to build, check the ESP-IDF installation and version
- If models fail to convert, check the model paths and formats
- For GUI issues, check Python dependencies and ensure all required packages are installed

## Next Steps

After completing this setup guide, refer to the User Manual for detailed instructions on using the system for damage detection in various scenarios.
