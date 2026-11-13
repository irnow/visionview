import cv2
import importlib
import os
from pathlib import Path

_MODEL_CACHE = {}

PARAMETERS = {
    "model_script": {
        "type": "ai_modules", 
        "options": ["None"],
        "default": "rfdetr_bbox"
    },
    "model_dir": {
        "type": "dir",
        "default": ""
    },
    "resolution": {
        "type": "enum",
        "options": ['Original', '320', '640', '800', '1024', '1280'],
        "default": '640'
    },
    "conf_thresh": {
        "type": "float",
        "min": 0.0, "max": 1.0, "default": 0.35, "step": 0.05
    },
    "device": {
        "type": "enum",
        "options": ["Auto (GPU/CPU)", "CPU", "CUDA:0"],
        "default": "Auto (GPU/CPU)"
    }
}

def get_ai_scripts():
    """Helper to find all scripts in the ai_modules folder"""
    import sys
    ai_path = Path(__file__).resolve().parent.parent / "ai_modules"
    if not ai_path.exists() and getattr(sys, 'frozen', False):
        base = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
        if (base / "ai_modules").exists():
            ai_path = base / "ai_modules"

    if not ai_path.exists():
        return ["rfdetr_bbox", "rfdetr_segmentation"]
    
    ai_module_list = list({
        f.stem for f in ai_path.iterdir() 
        if f.suffix in [".py", ".pyc"] and f.stem not in ["__init__", "desktop"]
    })
    return sorted(ai_module_list) if ai_module_list else ["rfdetr_bbox", "rfdetr_segmentation"]

def resolve_model_files(model_dir):
    """
    Finds the model weights file and metadata JSON from a directory path or direct file path.
    Supports both Schema A (training_metadata.json) and Schema B (metadata.json).
    """
    if not model_dir or not os.path.exists(model_dir):
        return None, None

    # Handle if direct file was selected
    if os.path.isfile(model_dir):
        if model_dir.lower().endswith(('.pth', '.pt', '.bin', '.onnx')):
            model_path = model_dir
            search_dir = os.path.dirname(model_dir)
        else:
            search_dir = os.path.dirname(model_dir)
            model_path = None
    else:
        search_dir = model_dir
        model_path = None

    # Search for weights if not directly given
    if not model_path:
        for fname in ["checkpoint_best_total.pth", "best_rfdetr_weights.pt", "best.pth", "best.pt", "model.pth"]:
            cand = os.path.join(search_dir, fname)
            if os.path.isfile(cand):
                model_path = cand
                break

        if not model_path:
            # Fallback: scan search_dir for any .pth or .pt file
            for root, _, files in os.walk(search_dir):
                for f in files:
                    if f.lower().endswith(('.pth', '.pt')):
                        model_path = os.path.join(root, f)
                        break
                if model_path:
                    break

    # Search for metadata
    metadata_path = None
    for mname in ["training_metadata.json", "metadata.json", "dataset.json", "config.json"]:
        cand = os.path.join(search_dir, mname)
        if os.path.isfile(cand):
            metadata_path = cand
            break
        # Also check parent directory
        parent_cand = os.path.join(os.path.dirname(search_dir), mname)
        if os.path.isfile(parent_cand):
            metadata_path = parent_cand
            break

    return model_path, metadata_path

def process(frame, params=None):
    if frame is None:
        return None
    if params is None:
        params = {}

    from utils.device_compatibility import check_deep_learning_support
    supported, _, _ = check_deep_learning_support()
    if not supported:
        return frame

    script_name = params.get("model_script", "None")
    model_dir = params.get("model_dir", "")

    if script_name == "None" or not model_dir or not os.path.exists(model_dir): 
        return frame
    
    model_path, metadata_path = resolve_model_files(model_dir)
    if not model_path or not os.path.exists(model_path):
        return frame

    res_str = str(params.get("resolution", '640'))
    target_size = None if res_str == "Original" else (int(res_str) if res_str.isdigit() else 640)
    conf = float(params.get("conf_thresh", 0.35))
    device = params.get("device", "Auto (GPU/CPU)")

    cache_key = f"{script_name}_{model_path}_{device}"

    try:
        module = importlib.import_module(f"ai_modules.{script_name}")
        
        # Check cache using unique key
        if cache_key not in _MODEL_CACHE:
            print(f"{'-'*5} INITIALIZING MODEL: {cache_key} {'-'*5}")
            if hasattr(module, 'MODEL'):
                try:
                    _MODEL_CACHE[cache_key] = module.MODEL(metadata_path, model_path, device=device)
                except TypeError:
                    # Fallback for models without device argument in __init__
                    _MODEL_CACHE[cache_key] = module.MODEL(metadata_path, model_path)
            else:
                return frame

        model_wrapper = _MODEL_CACHE[cache_key]

        # Call predict with target_size if supported, else standard predict
        try:
            return model_wrapper.predict(frame, conf_thresh=conf, target_size=target_size)
        except TypeError:
            return model_wrapper.predict(frame, conf_thresh=conf)

    except Exception as e:
        print(f"Deep Learning Error ({script_name}): {e}")
        return frame
