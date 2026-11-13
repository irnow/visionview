#!/usr/bin/env python3
"""
==============================================================================
VisionView - RF-DETR Object Detection AI Module
Supports both standalone CLI execution and VisionView live pipeline integration
via the standard `MODEL` class interface.
==============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import rfdetr
    HAS_RFDETR = True
except ImportError:
    HAS_RFDETR = False

DEFAULT_COLORS = [
    (16, 185, 129),   # Emerald
    (37, 99, 235),    # Blue
    (245, 158, 11),   # Amber
    (139, 92, 246),   # Purple
    (239, 68, 68),    # Red
    (6, 182, 212),    # Cyan
    (236, 72, 153),   # Pink
    (249, 115, 22)    # Orange
]

def load_metadata_config(metadata_path=None):
    """
    Loads model training metadata or configuration from path or common fallback locations.
    Supports both Schema A (classes, model_architecture) and Schema B (dataset.class_names, model.model_type).
    """
    meta_paths = []
    if metadata_path:
        meta_paths.append(metadata_path)
        if os.path.isdir(metadata_path):
            meta_paths.append(os.path.join(metadata_path, "training_metadata.json"))
            meta_paths.append(os.path.join(metadata_path, "metadata.json"))
        else:
            meta_paths.append(os.path.join(os.path.dirname(metadata_path), "training_metadata.json"))
            meta_paths.append(os.path.join(os.path.dirname(metadata_path), "metadata.json"))

    meta_paths.extend([
        "training_metadata.json",
        "metadata.json",
        os.path.join("weights", "..", "training_metadata.json"),
        os.path.join("weights", "..", "metadata.json")
    ])

    for mp in meta_paths:
        if mp and os.path.isfile(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"[Metadata] Loaded configuration from {mp}")
                    return data
            except Exception as e:
                print(f"[Metadata] Warning: Failed reading {mp}: {e}")
    return {}

def resolve_model_class(arch_name: str):
    """Dynamically resolves the RF-DETR model class by name or alias"""
    if not HAS_RFDETR:
        return None

    if hasattr(rfdetr, arch_name):
        return getattr(rfdetr, arch_name)
    for attr in dir(rfdetr):
        if attr.lower() == arch_name.lower():
            return getattr(rfdetr, attr)

    arch_lower = arch_name.lower()
    if "nano" in arch_lower and hasattr(rfdetr, "RFDETRNano"):
        return getattr(rfdetr, "RFDETRNano")
    if "small" in arch_lower and hasattr(rfdetr, "RFDETRSmall"):
        return getattr(rfdetr, "RFDETRSmall")
    if "base" in arch_lower and hasattr(rfdetr, "RFDETRBase"):
        return getattr(rfdetr, "RFDETRBase")
    if "large" in arch_lower and hasattr(rfdetr, "RFDETRLarge"):
        return getattr(rfdetr, "RFDETRLarge")
    if "medium" in arch_lower and hasattr(rfdetr, "RFDETRMedium"):
        return getattr(rfdetr, "RFDETRMedium")
    return getattr(rfdetr, "RFDETRSmall", None)

def resolve_default_weights(weights_path: str) -> str:
    """Finds best matching model weights file from path or search heuristics"""
    if weights_path and os.path.isfile(weights_path):
        return os.path.abspath(weights_path)
    
    search_dirs = []
    if weights_path and os.path.isdir(weights_path):
        search_dirs.append(weights_path)
        search_dirs.append(os.path.join(weights_path, "weights"))
    search_dirs.extend([".", "weights"])

    candidates = [
        "checkpoint_best_total.pth",
        "best_rfdetr_weights.pt",
        "best.pth",
        "best.pt",
        "model.pth"
    ]

    for s_dir in search_dirs:
        for c in candidates:
            p = os.path.join(s_dir, c)
            if os.path.isfile(p):
                print(f"[Auto-Detect] Using discovered checkpoint: {p}")
                return os.path.abspath(p)

    return os.path.abspath(weights_path) if weights_path else ""


class MODEL:
    """
    Standard VisionView Deep Learning Module Interface for RF-DETR Object Detection.
    Instantiated by computer_vision.deep_learning when selected in the process canvas.
    """
    def __init__(self, metadata_path=None, model_path=None, device="Auto (GPU/CPU)"):
        if not HAS_RFDETR:
            raise ImportError("The 'rfdetr' package is not installed. Please install rfdetr (e.g. pip install -r requirements-ai.txt).")

        self.metadata = load_metadata_config(metadata_path)
        
        # Parse architecture
        self.model_arch = (
            self.metadata.get("model_architecture") or 
            self.metadata.get("model", {}).get("model_type") or 
            "RFDETRSmall"
        )

        # Parse classes
        self.classes = (
            self.metadata.get("classes") or 
            self.metadata.get("class_names") or 
            self.metadata.get("dataset", {}).get("class_names", [])
        )
        self.num_classes = self.metadata.get("num_classes", len(self.classes) if self.classes else None)

        # Parse custom colors or fallback to default
        raw_colors = self.metadata.get("colors") or self.metadata.get("dataset", {}).get("colors", [])
        if raw_colors and isinstance(raw_colors, list):
            self.colors = raw_colors
        else:
            self.colors = DEFAULT_COLORS

        # Resolve device
        dev_str = str(device).lower() if device else "auto"
        if dev_str.startswith("cuda") and HAS_TORCH and torch.cuda.is_available():
            self.device = "cuda"
        elif dev_str.startswith("cpu"):
            self.device = "cpu"
        else:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        # Resolve weights
        resolved_weights = resolve_default_weights(model_path)
        if not os.path.isfile(resolved_weights):
            raise FileNotFoundError(f"RF-DETR weights not found at: {model_path} (resolved: {resolved_weights})")

        print(f"[RF-DETR BBox] Initializing {self.model_arch} (classes={self.num_classes}) on {self.device} with {resolved_weights}...")
        model_cls = resolve_model_class(self.model_arch)
        if model_cls is None:
            raise ValueError(f"Could not resolve RF-DETR model class for architecture: {self.model_arch}")

        if self.num_classes is not None and self.num_classes > 0:
            self.model = model_cls(num_classes=self.num_classes, pretrain_weights=resolved_weights)
        else:
            self.model = model_cls(pretrain_weights=resolved_weights)

        if hasattr(self.model, "optimize_for_inference"):
            try:
                self.model.optimize_for_inference()
            except Exception as e:
                print(f"[RF-DETR BBox] Notice: optimize_for_inference skipped: {e}")

    def predict(self, frame, conf_thresh=0.35, target_size=None):
        """
        Inference on an image frame (numpy array, PIL Image, or file path).
        Returns annotated BGR numpy ndarray for real-time display in the pipeline.
        """
        if frame is None:
            return None

        # 1. Normalize image input
        if isinstance(frame, np.ndarray):
            orig_bgr = frame
            orig_h, orig_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        elif isinstance(frame, Image.Image):
            pil_img = frame.convert("RGB")
            orig_w, orig_h = pil_img.size
            orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(frame, (str, Path)):
            pil_img = Image.open(str(frame)).convert("RGB")
            orig_w, orig_h = pil_img.size
            orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            return frame

        # 2. Handle optional model resolution scaling for acceleration
        scale_ratio = 1.0
        if target_size and str(target_size).isdigit():
            target_width = int(target_size)
            if target_width > 0 and orig_w > 0 and target_width != orig_w:
                scale_ratio = target_width / float(orig_w)
                target_height = max(1, int(float(orig_h) * scale_ratio))
                inference_img = pil_img.resize((target_width, target_height), Image.Resampling.BILINEAR)
            else:
                inference_img = pil_img
        else:
            inference_img = pil_img

        # 3. Model Prediction
        try:
            preds = self.model.predict(images=inference_img, threshold=conf_thresh, device=self.device)
        except Exception as e:
            print(f"[RF-DETR BBox] Prediction error: {e}")
            return orig_bgr

        raw_boxes = getattr(preds, "xyxy", [])
        scores = getattr(preds, "confidence", [])
        pred_classes = getattr(preds, "class_id", [])

        if raw_boxes is None or len(raw_boxes) == 0:
            return orig_bgr

        # Rescale boxes back to original coordinates if downscaled
        if scale_ratio != 1.0:
            boxes = [[b[0] / scale_ratio, b[1] / scale_ratio, b[2] / scale_ratio, b[3] / scale_ratio] for b in raw_boxes]
        else:
            boxes = raw_boxes

        # 4. Fast OpenCV overlay rendering on native resolution
        annotated_bgr = orig_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for idx, box in enumerate(boxes):
            score = float(scores[idx]) if len(scores) > idx else 0.0
            cls_id = int(pred_classes[idx]) if len(pred_classes) > idx else 0
            cls_name = self.classes[cls_id] if (self.classes and 0 <= cls_id < len(self.classes)) else str(cls_id)

            rgb_col = self.colors[cls_id % len(self.colors)]
            bgr_col = (int(rgb_col[2]), int(rgb_col[1]), int(rgb_col[0]))

            bx1, by1, bx2, by2 = [int(v) for v in box]
            bx1 = max(0, min(orig_w - 1, bx1))
            by1 = max(0, min(orig_h - 1, by1))
            bx2 = max(0, min(orig_w - 1, bx2))
            by2 = max(0, min(orig_h - 1, by2))

            # Draw bounding box
            cv2.rectangle(annotated_bgr, (bx1, by1), (bx2, by2), bgr_col, 2, cv2.LINE_AA)

            # Draw label badge
            tag = f"{cls_name} {score:.2f}"
            (tw, th), baseline = cv2.getTextSize(tag, font, 0.45, 1)
            ty1 = max(0, by1 - th - 6)
            cv2.rectangle(annotated_bgr, (bx1, ty1), (bx1 + tw + 6, ty1 + th + 6), bgr_col, -1)
            cv2.putText(annotated_bgr, tag, (bx1 + 3, ty1 + th + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_bgr


# ------------------------------------------------------------------------------
# Standalone CLI and Visualization Support
# ------------------------------------------------------------------------------

def render_detection_overlay(img_path: str, boxes, scores, classes, class_names, output_path: str):
    """Preserved for CLI file-based visualization export"""
    raw_img = Image.open(img_path).convert("RGBA")
    w, h = raw_img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    colors = DEFAULT_COLORS

    for idx, box in enumerate(boxes):
        score = float(scores[idx]) if len(scores) > idx else 0.0
        cls_id = int(classes[idx]) if len(classes) > idx else 0
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        
        col = colors[cls_id % len(colors)]
        outline_col = (col[0], col[1], col[2], 255)

        bx1, by1, bx2, by2 = [int(v) for v in box]
        draw.rectangle([(bx1, by1), (bx2, by2)], outline=outline_col, width=2)

        tag = f"{cls_name} {score:.2f}"
        try:
            tb = font.getbbox(tag)
            tw, th = tb[2] - tb[0] + 10, tb[3] - tb[1] + 6
        except Exception:
            tw, th = len(tag) * 7 + 10, 18
        ty1 = max(0, by1 - th)
        draw.rectangle([(bx1, ty1), (bx1 + tw, ty1 + th)], fill=outline_col)
        draw.text((bx1 + 5, ty1 + 2), tag, fill=(255, 255, 255), font=font)

    final_img = Image.alpha_composite(raw_img, overlay).convert("RGB")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_img.save(output_path, "PNG")
    print(f"[Overlay] Saved visualization to: {output_path}")

def run_detection_inference(weights_path: str, source_path: str, conf_threshold: float = 0.35, device: str = "cuda", img_size: int = 1280):
    weights_path = resolve_default_weights(weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    model_wrapper = MODEL(model_path=weights_path, device=device)

    image_paths = []
    if os.path.isdir(source_path):
        image_paths = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [source_path]

    out_dir = "inference_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for img_p in image_paths:
        if not os.path.exists(img_p):
            print(f"[Warning] Image not found: {img_p}")
            continue

        print(f"[Inference] Processing {img_p}...")
        img_bgr = cv2.imread(img_p)
        annotated = model_wrapper.predict(img_bgr, conf_thresh=conf_threshold, target_size=img_size)
        out_path = os.path.join(out_dir, f"overlay_{Path(img_p).name}")
        cv2.imwrite(out_path, annotated)
        print(f" -> Output saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR Object Detection Inference")
    parser.add_argument("--weights", type=str, default="checkpoint_best_total.pth", help="Path to best checkpoint or weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or image directory")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--img_size", "--img-size", type=int, default=1280, help="Resize for prediction")
    args = parser.parse_args()

    run_detection_inference(args.weights, args.source, args.conf, args.device, args.img_size)
