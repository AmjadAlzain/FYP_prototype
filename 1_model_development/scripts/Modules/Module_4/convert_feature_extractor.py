# Module4/convert_feature_extractor.py
#!/usr/bin/env python3
"""
3-stage converter:
  1. PyTorch (INT8 + pruning)  → ONNX
  2. ONNX                      → TensorFlow SavedModel
  3. SavedModel                → fully-INT8 TFLite

Each stage is run in its own tiny virtual-env so that mutually
incompatible packages never touch the same interpreter.
"""

import os, sys, subprocess, shutil, textwrap
from pathlib import Path

# -------------------------------------------------------------------
# project paths / constants
# -------------------------------------------------------------------
THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT= THIS_FILE.parents[4]
SCRIPTS_ROOT= THIS_FILE.parents[2]
NUM_CLASSES = 5
IMG_SIZE    = 128
WIDTH_MULT  = 1.0

QAT_CKPT    = PROJECT_ROOT / "1_model_development/models/feature_extractor_quantized.pth"
OUT_TFLITE  = PROJECT_ROOT / "2_firmware_esp32/components/model_data/feature_extractor_model.tflite"
TMP_ONNX    = PROJECT_ROOT / "_tmp_fx.onnx"
TMP_SAVED   = PROJECT_ROOT / "_tmp_fx_tf"

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def venv_paths(venv):
    if os.name == "nt":
        return venv/"Scripts"/"python.exe", venv/"Scripts"/"pip.exe"
    return venv/"bin"/"python", venv/"bin"/"pip"

def ensure_venv(name: str, pkgs: list[str]) -> Path:
    venv_dir = PROJECT_ROOT / name
    if not venv_dir.exists():
        print(f"🛠  creating venv {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    py  = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python.exe"
    pip = lambda: [str(py), "-m", "pip"]

    # 1️⃣  bootstrap / upgrade pip **unconditionally**
    subprocess.check_call([str(py), "-m", "ensurepip", "--default-pip"])
    subprocess.check_call(pip() + ["install", "--upgrade", "-q", "pip"])

    # 2️⃣  install stage-specific wheels (idempotent: already-present → skipped)
    print("   ▶ installing into", venv_dir.name, ":", " ".join(pkgs))
    if pkgs:                       # no-op for the tiny tf-lite venv
        subprocess.check_call(pip() + ["install", "-q"] + pkgs)

    return py


def run_stage(python:Path, stage:str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_ROOT)
    subprocess.check_call([str(python), str(THIS_FILE), stage], env=env)

# ------------------------------------------------------------------
# 0)  Minimal ClassifierWrapper  (↑ put this near the top of file!)
# ------------------------------------------------------------------
from Modules.Module_2.tinynas_detection_model import TinyNASFeatureExtractor

class ClassifierWrapper(nn.Module):
    def __init__(self, width_mult, num_classes, img_size=128):
        super().__init__()
        self.fe = TinyNASFeatureExtractor(width_mult)
        with torch.no_grad():
            x = torch.randn(1, 3, img_size, img_size)
            feat_dim = self.fe(x).view(1, -1).size(1)
        self.proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True))

    def forward(self, x, *, return_features=False):
        x = self.fe(x).view(x.size(0), -1)
        feat = self.proj(x)
        return feat if return_features else feat


# -------------------------------------------------------------------
# stage functions ----------------------------------------------------
# -------------------------------------------------------------------
def stage_onnx():
    print("🔄 [ONNX] exporting …")

    # ---- 1. build the *same* skeleton that was used during QAT
    fx = ClassifierWrapper(WIDTH_MULT, NUM_CLASSES, IMG_SIZE)
    fx.fe = fx.feature_extractor          # ① give it the expected alias
    del fx.feature_extractor

    # put the model in QAT-ready state *before* loading weights
    fx.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(fx, inplace=True)

    # ---- 2. load the checkpoint  (ignore missing/extra keys – harmless)
    ckpt = torch.load(QAT_CKPT, map_location='cpu')
    fx.load_state_dict(ckpt, strict=False)

    # ---- 3. convert to a true INT8 graph
    fx_int8 = torch.quantization.convert(fx.eval(), inplace=False)

    # optional sparsity (unchanged)
    for m in fx_int8.modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, "weight", 0.50)
            prune.remove(m, "weight")

    # ---- 4. wrap to emit the 256-d feature vector only
    class FeatureOnly(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, x):     return self.core(x, return_features=True)

    feat = FeatureOnly(fx_int8).eval()

    with torch.no_grad():
        print("   ✅ shape:", feat(torch.randn(1,3,IMG_SIZE,IMG_SIZE)).shape)

    # ---- 5. export to ONNX (unchanged)
    dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
    torch.onnx.export(
        feat, dummy, str(TMP_ONNX),
        input_names=['input'], output_names=['output'],
        opset_version=13, do_constant_folding=True
    )
    print("   ✅ ONNX saved to", TMP_ONNX)


def stage_tf():
    print("🔄 [TF ] ONNX → SavedModel")
    import onnx
    from onnx_tf.backend import prepare
    model = onnx.load(str(TMP_ONNX))
    tf_rep = prepare(model)
    if TMP_SAVED.exists(): shutil.rmtree(TMP_SAVED)
    tf_rep.export_graph(str(TMP_SAVED))
    print("   ✅ SavedModel:", TMP_SAVED)

def stage_tflite():
    print("🔄 [TFL] SavedModel → INT8 TFLite")
    import tensorflow as tf, numpy as np
    conv = tf.lite.TFLiteConverter.from_saved_model(str(TMP_SAVED))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    dummy = np.zeros((1,3,IMG_SIZE,IMG_SIZE),np.float32)
    conv.representative_dataset = ([dummy] for _ in range(100))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    OUT_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TFLITE.write_bytes(conv.convert())
    print("   ✅ TFLite:", OUT_TFLITE)

# -------------------------------------------------------------------
# main orchestrator
# -------------------------------------------------------------------
def main():
    venv_cfg = {
        "onnx":   ("venv_onnx",   ["torch", "torchvision"]),
        "tf":     ("venv_tf",     ["onnx", "onnx-tf", "tensorflow==2.14.0", "keras==2.14.0"]),
        "tflite": ("venv_tflite", ["tensorflow==2.14.0"])
    }

    if len(sys.argv) == 1:
        # big orchestrator run
        for stage, (dir_name, pkgs) in venv_cfg.items():
            py = ensure_venv(dir_name, pkgs)
            run_stage(py, stage)

        # tidy up temp artefacts
        for p in (TMP_ONNX, TMP_SAVED):
            if p.exists():
                p.unlink() if p.is_file() else shutil.rmtree(p, ignore_errors=True)

        print("\n🎉  All done – INT8 model at", OUT_TFLITE)
        return

    # ----------------------------------------------------------------
    # sub-process execution:  argv[1] == stage
    # ----------------------------------------------------------------
    stage = sys.argv[1]
    if   stage == "onnx":   stage_onnx()
    elif stage == "tf":     stage_tf()
    elif stage == "tflite": stage_tflite()
    else:
        sys.exit(f"❌ unknown stage '{stage}'")

if __name__ == "__main__":
    main()
