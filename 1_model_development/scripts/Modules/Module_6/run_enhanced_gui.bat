@echo off
title Enhanced ESP32-S3-EYE Container Detection GUI

echo ========================================
echo Enhanced Container Detection System
echo Multi-Mode: ESP32 + Laptop + Video
echo ========================================
echo.

:: Check if virtual environment exists
if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo No virtual environment found. Using system Python...
)

:: Install/update requirements
echo Checking dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements_enhanced.txt

echo.
echo Starting Enhanced GUI...
echo.
echo Available Modes:
echo 1. ESP32-S3-EYE Mode - Hardware device communication
echo 2. Laptop Camera Mode - Real-time webcam inference
echo 3. Video Upload Mode - Batch processing of video files
echo 4. Analytics Dashboard - Detection statistics and history
echo.

:: Launch the enhanced GUI
python container_detection_gui_enhanced.py

echo.
echo GUI closed. Press any key to exit...
pause > nul
