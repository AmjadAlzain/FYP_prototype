"""
esp32_deploy.py - Simplified ESP32-S3-EYE deployment script
"""
import os
import subprocess
import shutil
from pathlib import Path
import numpy as np

def create_header_files():
    """Create necessary header files for ESP32 deployment"""
    print("Creating model header files...")
    
    # 1. Create TFLite model header
    os.makedirs("components/tflite_micro", exist_ok=True)
    with open("components/tflite_micro/model_data.c", 'w') as f:
        f.write("/**\n")
        f.write(" * model_data.c - Model data for ESP32-S3-EYE\n")
        f.write(" */\n\n")
        f.write("#include \"model_data.h\"\n\n")
        f.write("// Simplified model data for testing\n")
        f.write("const unsigned char model_tflite[] = {\n")
        f.write("  0x54, 0x46, 0x4c, 0x33, // TFL3 header\n")
        f.write("  // Remaining data omitted for brevity\n")
        f.write("};\n\n")
        f.write("const unsigned int model_tflite_len = 4;\n")
    
    # 2. Create EmbHD headers
    os.makedirs("components/embhd/headers", exist_ok=True)
    
    # Create projection header
    with open("components/embhd/headers/embhd_projection.h", 'w') as f:
        f.write("#ifndef EMBHD_PROJECTION_H\n")
        f.write("#define EMBHD_PROJECTION_H\n\n")
        f.write("#include \"embhd.h\"\n\n")
        f.write("#define EMBHD_PROJECTION_DIM 10000\n")
        f.write("#define EMBHD_FEATURE_DIM 1280\n\n")
        f.write("// Simple projection data for testing\n")
        f.write("const int8_t embhd_projection[EMBHD_PROJECTION_DIM][EMBHD_FEATURE_DIM] = {{0}};\n\n")
        f.write("#endif /* EMBHD_PROJECTION_H */\n")
    
    # Create prototypes header
    with open("components/embhd/headers/embhd_prototypes.h", 'w') as f:
        f.write("#ifndef EMBHD_PROTOTYPES_H\n")
        f.write("#define EMBHD_PROTOTYPES_H\n\n")
        f.write("#include \"embhd.h\"\n\n")
        f.write("#define EMBHD_HD_DIM 10000\n")
        f.write("#define EMBHD_TEST_CLASSES 5\n\n")
        f.write("// Simple prototype data for testing\n")
        f.write("const int8_t embhd_prototypes[EMBHD_TEST_CLASSES][EMBHD_HD_DIM] = {{0}};\n\n")
        f.write("#endif /* EMBHD_PROTOTYPES_H */\n")
    
    print("Header files created successfully")
    return True

def flash_esp32():
    """Build and flash firmware to ESP32-S3-EYE on COM3"""
    print("Building and flashing firmware to ESP32-S3-EYE...")
    
    try:
        # Build the firmware
        build_cmd = 'idf.py build'
        subprocess.run(build_cmd, shell=True, check=True)
        
        # Flash the firmware to COM3
        flash_cmd = 'idf.py -p COM3 flash'
        subprocess.run(flash_cmd, shell=True, check=True)
        
        print("Firmware flashed successfully to ESP32-S3-EYE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during build/flash: {e}")
        return False

def main():
    """Main deployment function"""
    print("=== ESP32-S3-EYE Deployment ===")
    
    # Create header files
    if not create_header_files():
        print("Error creating header files. Aborting.")
        return
    
    # Build and flash firmware
    if not flash_esp32():
        print("Error flashing firmware. Deployment incomplete.")
        return
    
    print("=== Deployment completed successfully ===")
    print("The ESP32-S3-EYE is now running with the test models")
    print("You can use the Python GUI to interact with it")

if __name__ == "__main__":
    main()
