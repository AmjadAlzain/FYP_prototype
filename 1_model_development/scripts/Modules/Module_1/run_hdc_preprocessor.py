#!/usr/bin/env python3
"""
Run script for HDC Data Preprocessor
Creates prototype/hdc_dataset/ from SeaFront container images
"""

import sys
from pathlib import Path

# Add project paths
BASE = Path(__file__).resolve().parents[4]
sys.path.append(str(BASE))
sys.path.append(str(BASE / "1_model_development" / "scripts"))

from hdc_data_preprocessor import HDCDataPreprocessor

def main():
    """Run HDC data preprocessing"""
    print("🚀 HDC Dataset Creation Tool")
    print("=" * 40)
    print("Creating whole container images dataset for HDC classifier")
    print("Source: SeaFront v1.0.0 Synthetic Container Dataset")
    print("Output: prototype/hdc_dataset/")
    print()
    
    # Check if source datasets exist
    seafront_root = BASE / "SeaFront_v1_0_0" / "SeaFront_v1_0_0"
    seafront_test = BASE / "SeaFront_v1_0_0_TEST"
    
    if not seafront_root.exists():
        print(f"❌ SeaFront training data not found at: {seafront_root}")
        print("Please ensure SeaFront_v1_0_0 dataset is extracted to project root")
        return
    
    if not seafront_test.exists():
        print(f"⚠️ SeaFront test data not found at: {seafront_test}")
        print("Test split will be skipped")
    
    print("📁 Source datasets found, starting preprocessing...")
    
    # Create and run preprocessor
    preprocessor = HDCDataPreprocessor(
        output_size=256,  # Standard size for HDC/TinyNAS input
        min_container_size=50  # Skip very small containers
    )
    
    try:
        preprocessor.run()
        print("\n🎉 HDC dataset creation completed successfully!")
        print("\nNext steps:")
        print("1. Review dataset statistics in prototype/hdc_dataset/dataset_statistics.json")
        print("2. Update Module 3 HDC classifier to use prototype/hdc_dataset/")
        print("3. Train HDC classifier with whole container images")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("Please check the error details above and try again")
        return

if __name__ == "__main__":
    main()
