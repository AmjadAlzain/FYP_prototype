# Module_3/embhd/convert_torch_hd_to_embhd.py
"""
Convert a trained PyTorch HDC model to EmbHD format

This script takes a trained torch_hd model and its projection matrix
and converts it to EmbHD format for deployment on ESP32-S3-EYE.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse

# Add project root to path
BASE = Path(__file__).resolve().parents[3]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

# Import EmbHD
from Modules.Module_3.embhd.embhd_py import EmbHDModel, EmbHDVectorType

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch HDC model to EmbHD format')
    parser.add_argument('--model_path', type=str, default=BASE / "module3_hdc_model.pth",
                        help='Path to trained PyTorch HDC model')
    parser.add_argument('--projection_path', type=str, default=BASE / "module3_hdc_projection.pth",
                        help='Path to projection matrix')
    parser.add_argument('--output_path', type=str, default=BASE / "module3_hdc_model_embhd.pth",
                        help='Path to save EmbHD model')
    parser.add_argument('--headers_dir', type=str, default=BASE / "Modules" / "Module_3" / "embhd" / "headers",
                        help='Directory to save C headers')
    parser.add_argument('--hd_dim', type=int, default=10000,
                        help='Hyperdimensional vector dimension for EmbHD model')
    args = parser.parse_args()

    print(f"Converting PyTorch HDC model to EmbHD format...")
    print(f"  Model path: {args.model_path}")
    print(f"  Projection path: {args.projection_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Headers directory: {args.headers_dir}")
    print(f"  HD dimension: {args.hd_dim}")

    try:
        # Load PyTorch HDC model
        torch_hd_model = torch.load(args.model_path)
        if not hasattr(torch_hd_model, 'prototypes'):
            raise ValueError("Invalid torch_hd model format (no prototypes)")

        # Load projection matrix
        projection_matrix = torch.load(args.projection_path)
        if not isinstance(projection_matrix, torch.Tensor):
            raise ValueError("Invalid projection matrix format")

        print(f"Loaded PyTorch HDC model with {torch_hd_model.prototypes.shape[0]} classes")
        print(f"Loaded projection matrix with shape {projection_matrix.shape}")

        # Convert to EmbHD format
        embhd_model = EmbHDModel.from_torch_hd(torch_hd_model, projection_matrix, args.hd_dim)

        # Save EmbHD model
        embhd_model.save(args.output_path)
        print(f"Saved EmbHD model to {args.output_path}")

        # Export C headers
        proj_header, proto_header = embhd_model.export_c_headers(args.headers_dir)
        print(f"Exported C headers:")
        print(f"  - {proj_header}")
        print(f"  - {proto_header}")

        print("Conversion complete!")

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
