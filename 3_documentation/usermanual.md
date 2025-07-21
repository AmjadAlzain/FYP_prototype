# ESP32-S3-EYE Damage Detection System
## User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Hardware Requirements](#hardware-requirements)
4. [Getting Started](#getting-started)
5. [GUI Application](#gui-application)
6. [Using the System](#using-the-system)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## Introduction

The ESP32-S3-EYE Damage Detection System is a specialized computer vision solution for detecting and classifying damage in various environments, with a particular focus on maritime infrastructure. This user manual provides comprehensive instructions for setting up, operating, and maintaining the system.

## System Overview

The system consists of:

1. **ESP32-S3-EYE Hardware**: A compact, powerful AI camera module with an ESP32-S3 microcontroller
2. **Onboard AI Models**:
   - TinyNAS-optimized object detection model
   - Hyperdimensional Computing (HDC) classifier for damage classification
3. **PC-based GUI Application**: For viewing camera feed, managing detection settings, and recording results

The system performs real-time damage detection using a low-power, efficient architecture specifically optimized for edge deployment.

## Hardware Requirements

- ESP32-S3-EYE development board
- USB-C cable for power and communication
- PC with USB port running Windows, macOS, or Linux
- Python 3.7+ installed on PC

## Getting Started

### Connecting the ESP32-S3-EYE

1. Connect the ESP32-S3-EYE to your computer using the USB-C cable
2. The device should power on automatically, displaying the boot sequence on its LCD screen
3. Wait for the system initialization (approximately 5 seconds)

### Installing the GUI Application

1. Ensure Python 3.7 or newer is installed on your system
2. Install the required dependencies:
   ```
   pip install pyserial opencv-python pillow numpy matplotlib
   ```
3. Launch the GUI application:
   ```
   python Prototype/GUI/gui_app.py
   ```

## GUI Application

The GUI application provides a user-friendly interface for interacting with the ESP32-S3-EYE device.

### Main Interface Components

- **Camera Feed**: Displays real-time video from the ESP32-S3-EYE camera with detection overlays
- **Connection Settings**: Configure serial port and baudrate settings
- **Command Section**: Control training, testing, and recording functions
- **Log Display**: View system messages and detection information

### Connection Settings

- **Port**: Select the serial port for your ESP32-S3-EYE device (use "AUTO" for automatic detection)
- **Baud Rate**: Default is 115200 (do not change unless specifically instructed)
- **Connect Button**: Establish connection with the device

## Using the System

### Basic Operation

1. Launch the GUI application
2. Select the appropriate port (or use "AUTO" for automatic detection)
3. Click "Connect"
4. The camera feed should appear in the display area
5. Detection results will automatically overlay on the camera feed

### Training Mode

The system supports on-device training for fine-tuning the damage classifier:

1. Select a class number (0-4) from the dropdown menu
2. Click "Start Training"
3. Point the camera at examples of the damage class
4. Click "Stop Training" when sufficient examples have been captured

### Testing

To run the test suite and evaluate system performance:

1. Click "Run Test Suite" in the Commands section
2. View results in the log area

### Recording

To record video of detection results:

1. Click "Start Recording"
2. Perform detection operations
3. Click "Stop Recording"
4. Recordings are saved to a timestamped folder in the application directory

## Troubleshooting

### Device Not Detected

- Ensure the USB cable is securely connected
- Try a different USB port
- Select "AUTO" in the port selection to trigger automatic detection
- On Windows, check Device Manager for COM port conflicts

### No Camera Feed

- Disconnect and reconnect the device
- Restart the GUI application
- Check the log area for error messages

### Poor Detection Performance

- Ensure adequate lighting in the environment
- Hold the camera steady during detection
- Make sure the object is within the camera's field of view
- Consider retraining the damage classifier for your specific conditions

## Maintenance

### Firmware Updates

To update the device firmware:

1. Close the GUI application
2. Run the deployment script:
   ```
   python deploy_to_esp32.py
   ```
3. Follow the on-screen instructions

### Cleaning

- Gently clean the camera lens with a microfiber cloth
- Keep the ESP32-S3-EYE board free of dust and moisture

For additional support, please contact the system administrator or refer to the technical documentation.
