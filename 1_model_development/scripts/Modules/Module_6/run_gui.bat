@echo off
echo ========================================
echo ESP32-S3-EYE Container Detection GUI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Navigate to script directory
cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM Check if requirements are installed
echo Checking requirements...
python -c "import PyQt6; import cv2; import serial; print('All requirements satisfied')" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM Check if GUI file exists
if not exist "container_detection_gui.py" (
    echo ERROR: GUI file not found
    echo Expected: container_detection_gui.py
    pause
    exit /b 1
)

echo Starting ESP32-S3-EYE Container Detection GUI...
echo.
echo Instructions:
echo 1. Connect your ESP32-S3-EYE via USB
echo 2. Select the correct COM port in the GUI
echo 3. Click Connect to start monitoring
echo.
echo Press Ctrl+C to exit the application
echo.

REM Run the GUI application
python container_detection_gui.py

echo.
echo GUI application closed.
pause
