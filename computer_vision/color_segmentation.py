import cv2
import numpy as np

PARAMETERS = {
    "enable_segmentation": {
        "type": "bool",
        "default": False
    },
    "pick_color": {
        "type": "color",
        "default": "#00FF00"
    },
    "global_tolerance": {
        "type": "int",
        "min": 1, "max": 100,
        "default": 25,
        "step": 1
    },
    "tolerance_hue": {
        "type": "int",
        "min": 1, "max": 90,
        "default": 18,
        "step": 1
    },
    "tolerance_sat": {
        "type": "int",
        "min": 1, "max": 255,
        "default": 65,
        "step": 5
    },
    "tolerance_val": {
        "type": "int",
        "min": 1, "max": 255,
        "default": 65,
        "step": 5
    },
    "output_mode": {
        "type": "enum",
        "options": ["Masked Image", "Binary Mask", "Inverse Mask", "Color Highlight", "Contour Overlay"],
        "default": "Masked Image"
    },
    "clean_noise": {
        "type": "bool",
        "default": True
    },
    "manual_override": {
        "type": "bool",
        "default": False
    },
    "h_min": {
        "type": "int",
        "min": 0, "max": 179,
        "default": 35
    },
    "h_max": {
        "type": "int",
        "min": 0, "max": 179,
        "default": 85
    },
    "s_min": {
        "type": "int",
        "min": 0, "max": 255,
        "default": 50
    },
    "s_max": {
        "type": "int",
        "min": 0, "max": 255,
        "default": 255
    },
    "v_min": {
        "type": "int",
        "min": 0, "max": 255,
        "default": 50
    },
    "v_max": {
        "type": "int",
        "min": 0, "max": 255,
        "default": 255
    }
}

def hex_to_bgr(hex_code: str):
    """Converts a hex color string (#RRGGBB) to BGR tuple"""
    cleaned = hex_code.lstrip("#")
    if len(cleaned) == 6:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return (b, g, r)
    return (0, 255, 0)

def process(frame, params=None):
    """
    Color Range Segmentation Module with Visual Color Picker and Multi-Mode Output.
    """
    if frame is None:
        return None
    if params is None:
        params = {}

    enabled = bool(params.get("enable_segmentation", False))
    if not enabled:
        return frame

    manual = bool(params.get("manual_override", False))
    output_mode = params.get("output_mode", "Masked Image")
    clean_noise = bool(params.get("clean_noise", True))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if manual:
        h_min = int(params.get("h_min", 35))
        h_max = int(params.get("h_max", 85))
        s_min = int(params.get("s_min", 50))
        s_max = int(params.get("s_max", 255))
        v_min = int(params.get("v_min", 50))
        v_max = int(params.get("v_max", 255))

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Compute HSV range from picked color + tolerance
        hex_color = str(params.get("pick_color", "#00FF00"))
        b, g, r = hex_to_bgr(hex_color)

        target_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        th, ts, tv = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])

        tol_h = int(params.get("tolerance_hue", 18))
        tol_s = int(params.get("tolerance_sat", 65))
        tol_v = int(params.get("tolerance_val", 65))

        s_low = max(0, ts - tol_s)
        s_high = min(255, ts + tol_s)
        v_low = max(0, tv - tol_v)
        v_high = min(255, tv + tol_v)

        h_low = th - tol_h
        h_high = th + tol_h

        # Handle Hue boundary wrap-around (especially critical for Reds around 0/180)
        if h_low < 0:
            mask1 = cv2.inRange(hsv, np.array([0, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8))
            mask2 = cv2.inRange(hsv, np.array([180 + h_low, s_low, v_low], dtype=np.uint8), np.array([179, s_high, v_high], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)
        elif h_high > 179:
            mask1 = cv2.inRange(hsv, np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([179, s_high, v_high], dtype=np.uint8))
            mask2 = cv2.inRange(hsv, np.array([0, s_low, v_low], dtype=np.uint8), np.array([h_high - 180, s_high, v_high], dtype=np.uint8))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8))

    # Morphological noise removal
    if clean_noise:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Format output based on selected mode
    if output_mode == "Binary Mask":
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif output_mode == "Inverse Mask":
        return cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)
    elif output_mode == "Color Highlight":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        dimmed_gray = (gray_bgr.astype(np.float32) * 0.45).astype(np.uint8)
        return np.where(mask[:, :, None] == 255, frame, dimmed_gray)
    elif output_mode == "Contour Overlay":
        output = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        return output
    else: # Default: "Masked Image"
        return cv2.bitwise_and(frame, frame, mask=mask)
