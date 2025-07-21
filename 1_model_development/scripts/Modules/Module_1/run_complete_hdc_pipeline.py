#!/usr/bin/env python3
"""
Complete HDC Pipeline Runner
1. Creates HDC dataset from SeaFront whole container images
2. Trains HDC classifier on whole containers instead of patches
3. Generates benchmarks and reports
"""

import sys
import subprocess
from pathlib import Path
import time
import json

# Add project paths
BASE = Path(__file__).resolve().parents[4]
sys.path.append(str(BASE))
sys.path.append(str(BASE / "1_model_development" / "scripts"))

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False

def check_prerequisites():
    """Check if all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check SeaFront dataset
    seafront_root = BASE / "SeaFront_v1_0_0" / "SeaFront_v1_0_0"
    seafront_test = BASE / "SeaFront_v1_0_0_TEST"
    
    if not seafront_root.exists():
        print(f"❌ SeaFront training data not found at: {seafront_root}")
        print("Please extract SeaFront_v1_0_0 dataset to project root")
        return False
    
    if not seafront_test.exists():
        print(f"⚠️ SeaFront test data not found at: {seafront_test}")
        print("Test split will be skipped")
    
    # Check feature extractor
    models_dir = BASE / "1_model_development" / "models"
    feature_extractor = models_dir / "feature_extractor_fp32_best.pth"
    
    if not feature_extractor.exists():
        print(f"⚠️ Feature extractor not found at: {feature_extractor}")
        print("Placeholder features will be used (train Module 2 first for best results)")
    
    print("✅ Prerequisites check completed")
    return True

def main():
    """Run complete HDC pipeline"""
    print("🎯 Complete HDC Pipeline for Container Damage Classification")
    print("=" * 70)
    print("This pipeline will:")
    print("1. 📁 Create HDC dataset from SeaFront whole container images")
    print("2. 🧠 Train HDC classifier on whole containers (not patches)")
    print("3. 📊 Generate benchmarks and accuracy reports")
    print("4. 💾 Save models for deployment")
    print()
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return
    
    # Step 1: Create HDC dataset
    print("\n" + "="*70)
    print("STEP 1: Creating HDC Dataset from Whole Container Images")
    print("="*70)
    
    preprocessor_script = BASE / "1_model_development" / "scripts" / "Modules" / "Module_1" / "hdc_data_preprocessor.py"
    
    success = run_command(
        [sys.executable, str(preprocessor_script)],
        "HDC Dataset Creation"
    )
    
    if not success:
        print("❌ HDC dataset creation failed. Stopping pipeline.")
        return
    
    # Check if dataset was created
    hdc_dataset_root = BASE / "prototype" / "hdc_dataset"
    if not hdc_dataset_root.exists():
        print("❌ HDC dataset was not created. Stopping pipeline.")
        return
    
    # Load and display dataset statistics
    stats_file = hdc_dataset_root / "dataset_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print("\n📊 Dataset Statistics:")
        total_containers = 0
        for split in ['train', 'val', 'test']:
            if split in stats:
                split_total = sum(stats[split].values())
                total_containers += split_total
                print(f"  {split.upper()}: {split_total} containers")
        
        print(f"  TOTAL: {total_containers} containers")
    
    # Step 2: Train HDC classifier
    print("\n" + "="*70)
    print("STEP 2: Training HDC Classifier on Whole Container Images")
    print("="*70)
    
    training_script = BASE / "1_model_development" / "scripts" / "Modules" / "Module_3" / "train_hdc_whole_containers.py"
    
    # Train with balanced weighting (recommended)
    success = run_command(
        [sys.executable, str(training_script), "--weight-strategy", "balanced", "--epochs", "50"],
        "HDC Training (Balanced Weighting)"
    )
    
    if not success:
        print("❌ HDC training failed. Check error messages above.")
        return
    
    # Step 3: Display results
    print("\n" + "="*70)
    print("STEP 3: Results Summary")
    print("="*70)
    
    # Check for trained model
    models_dir = BASE / "1_model_development" / "models"
    hdc_model = models_dir / "module3_hdc_whole_containers_2048_balanced.npz"
    
    if hdc_model.exists():
        print(f"✅ HDC model saved: {hdc_model}")
        model_size = hdc_model.stat().st_size / (1024 * 1024)  # MB
        print(f"📏 Model size: {model_size:.2f} MB")
    
    # Check benchmarks
    benchmarks_dir = BASE / "1_model_development" / "benchmarks" / "hdc_whole_containers_2048_balanced"
    if benchmarks_dir.exists():
        print(f"📊 Benchmarks saved: {benchmarks_dir}")
        
        # Load training history if available
        history_file = benchmarks_dir / "training_history.csv"
        if history_file.exists():
            import pandas as pd
            history = pd.read_csv(history_file)
            final_val_acc = history['val_acc'].iloc[-1]
            print(f"🎯 Final validation accuracy: {final_val_acc:.2f}%")
    
    # Step 4: Next steps
    print("\n" + "="*70)
    print("STEP 4: Next Steps & Integration")
    print("="*70)
    
    print("✅ HDC Pipeline completed successfully!")
    print("\nIntegration with Enhanced GUI:")
    print("1. Update local_inference.py to use new HDC model")
    print("2. Test whole container classification in GUI")
    print("3. Deploy to ESP32 if needed")
    
    print(f"\nFiles created:")
    print(f"📁 Dataset: {hdc_dataset_root}")
    print(f"🧠 Model: {hdc_model}")
    print(f"📊 Benchmarks: {benchmarks_dir}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total pipeline time: {total_time/60:.1f} minutes")
    
    print("\n🎉 Complete HDC pipeline finished successfully!")

if __name__ == "__main__":
    main()
