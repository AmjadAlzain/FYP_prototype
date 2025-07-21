@echo off
title Module 6 Comprehensive Testing Suite

echo ================================================================
echo MODULE 6 COMPREHENSIVE TESTING SUITE
echo Enhanced ESP32-S3-EYE Container Detection System
echo ================================================================
echo.

:: Check if virtual environment exists
if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate
    echo.
) else (
    echo No virtual environment found. Using system Python...
    echo.
)

:: Install/update testing dependencies
echo Installing/updating testing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements_enhanced.txt
python -m pip install pytest pytest-cov coverage

echo.
echo ================================================================
echo STARTING COMPREHENSIVE TEST EXECUTION
echo ================================================================
echo.

:: Run all tests
python run_all_tests.py

echo.
echo ================================================================
echo INDIVIDUAL TEST EXECUTION OPTIONS
echo ================================================================
echo.

:menu
echo Choose test type to run individually:
echo.
echo [1] Unit Tests Only
echo [2] Integration Tests Only  
echo [3] A/B Tests Only
echo [4] Run All Tests Again
echo [5] Inspect Models
echo [6] Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto unit_tests
if "%choice%"=="2" goto integration_tests
if "%choice%"=="3" goto ab_tests
if "%choice%"=="4" goto all_tests
if "%choice%"=="5" goto inspect_models
if "%choice%"=="6" goto exit
goto menu

:unit_tests
echo.
echo Running Unit Tests...
echo ----------------------------------------------------------------
python test_unit.py
echo.
pause
goto menu

:integration_tests
echo.
echo Running Integration Tests...
echo ----------------------------------------------------------------
python test_integration.py
echo.
pause
goto menu

:ab_tests
echo.
echo Running A/B Tests...
echo ----------------------------------------------------------------
python test_ab.py
echo.
pause
goto menu

:all_tests
echo.
echo Running All Tests...
echo ----------------------------------------------------------------
python run_all_tests.py
echo.
pause
goto menu

:inspect_models
echo.
echo Inspecting Model Files...
echo ----------------------------------------------------------------
python model_inspector.py
echo.
pause
goto menu

:exit
echo.
echo ================================================================
echo TESTING COMPLETE
echo ================================================================
echo.
echo Check the generated test reports for detailed results.
echo Test reports are saved as JSON files with timestamps.
echo.
echo Thank you for using Module 6 Testing Suite!
echo.
pause
