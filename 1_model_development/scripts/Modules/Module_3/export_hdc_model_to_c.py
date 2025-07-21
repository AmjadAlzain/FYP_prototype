"""
Module 3: Export torchhd model to C headers
"""

import sys
import torch
import numpy as np
from pathlib import Path

# --- Path Setup ---
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
PROJECT_ROOT = SCRIPTS_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Add the 'Modules' directory to sys.path so that 'Modules.Module_2.train_classifier' can be imported
if str(SCRIPTS_ROOT.parent) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT.parent))
if str(SCRIPTS_ROOT.parent.parent) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT.parent.parent))

# IMPORT YOUR CUSTOM TorchHDCModel from train_hdc.py
try:
    from Modules.Module_3.train_hdc import TorchHDCModel # This is your custom model class
except ImportError:
    print("Error: Could not import TorchHDCModel from Modules.Module_3.train_hdc. Please ensure the path is correct.")
    sys.exit(1)


# --- Configuration ---
HDC_DIMENSIONS = 2048
MODELS_DIR = PROJECT_ROOT / "1_model_development" / "models"
# NOTE: This script exports to a component named "embhd", ensure this matches your project.
# The C code fixes were based on a component named "hdc".
OUTPUT_C_DIR = PROJECT_ROOT / "2_firmware_esp32" / "components" / "embhd"

# Container damage classes
DAMAGE_CLASSES = ['axis', 'concave', 'dentado', 'no_damage' , 'perforation']
NUM_CLASSES = len(DAMAGE_CLASSES)

def save_array_to_c_header(arr: np.ndarray, filename: Path, base_var_name: str, type_name: str = "int8_t"):
    """Saves a numpy array to a C header file defining an initializer macro and dimensions."""
    macro_name = f"{base_var_name.upper()}_INITIALIZER"
    
    dims = arr.shape
    
    with open(filename, 'w') as f:
        f.write(f"#ifndef {base_var_name.upper()}_H\n")
        f.write(f"#define {base_var_name.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")

        if len(dims) == 2:
            f.write(f"#define {base_var_name.upper()}_ROWS {dims[0]}\n")
            f.write(f"#define {base_var_name.upper()}_COLS {dims[1]}\n\n")
        elif len(dims) == 1:
            f.write(f"#define {base_var_name.upper()}_SIZE {dims[0]}\n\n")

        f.write(f"#define {macro_name} \\\n{{\n")

        flattened = arr.flatten().tolist()
        items_per_line = 16
        for i, val in enumerate(flattened):
            if i % items_per_line == 0:
                f.write("    ")
            f.write(f"{val}, ")
            if (i + 1) % items_per_line == 0 and (i + 1) != len(flattened):
                f.write("\\\n")
        
        if len(flattened) > 0:
            f.seek(f.tell() - 2)
            f.truncate()
            f.write(" ")
        
        f.write("\\\n}\n\n")
        f.write(f"#endif // {base_var_name.upper()}_H\n")
    print(f"Saved {base_var_name} (Shape: {arr.shape}) as {macro_name} to {filename}")

def main():
    print("=" * 60)
    print("Module 3: Export torchhd model to C headers")
    print("=" * 60)

    model_name = f"module3_torchhdc_model_{HDC_DIMENSIONS}_balanced.pth"
    HDC_MODEL_PATH = MODELS_DIR / model_name

    if not HDC_MODEL_PATH.exists():
        print(f"Error: HDC model not found at {HDC_MODEL_PATH}. Please train it first.")
        return

    print(f"\n🧠 Loading torchhd model from {HDC_MODEL_PATH}...")
    
    from Modules.Module_2.train_classifier import ClassifierWrapper, WIDTH_MULTIPLIER, IMG_SIZE
    dummy_device = torch.device("cpu") 

    with torch.no_grad():
        dummy_model = ClassifierWrapper(WIDTH_MULTIPLIER, NUM_CLASSES, IMG_SIZE).to(dummy_device)
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(dummy_device)
        dummy_features = dummy_model(dummy_input, return_features=True)
        FEATURE_SIZE = dummy_features.size(1)

    model = TorchHDCModel(FEATURE_SIZE, HDC_DIMENSIONS, NUM_CLASSES).to(dummy_device)
    model.load_state_dict(torch.load(HDC_MODEL_PATH, map_location=dummy_device))
    model.eval()

    # --- THIS IS THE CORRECTED LINE ---
    # PyTorch nn.Linear weights are already in the correct [out_features, in_features]
    # format needed by the C code. Transposing them was incorrect.
    projection_matrix = model.projection.weight.data.cpu().numpy()
    class_prototypes = model.class_prototypes.data.cpu().numpy()

    # Quantize to int8_t
    max_abs_proj_actual = np.max(np.abs(projection_matrix))
    if max_abs_proj_actual == 0: max_abs_proj_actual = 1.0
    scale_factor_proj = 127.0 / max_abs_proj_actual
    projection_matrix_int8 = np.round(projection_matrix * scale_factor_proj).clip(-128, 127).astype(np.int8)

    max_abs_proto_actual = np.max(np.abs(class_prototypes))
    if max_abs_proto_actual == 0: max_abs_proto_actual = 1.0
    scale_factor_proto = 127.0 / max_abs_proto_actual
    class_prototypes_int8 = np.round(class_prototypes * scale_factor_proto).clip(-128, 127).astype(np.int8)

    # Save to C header files
    OUTPUT_C_DIR.mkdir(parents=True, exist_ok=True)
    save_array_to_c_header(projection_matrix_int8, OUTPUT_C_DIR / "embhd_projection.h", "EMBHD_PROJ")
    save_array_to_c_header(class_prototypes_int8, OUTPUT_C_DIR / "embhd_prototypes.h", "EMBHD_PROTO")

    print("\n✅ HDC Model export to C headers complete!")

if __name__ == '__main__':
    main()