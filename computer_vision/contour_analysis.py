import cv2
import numpy as np

PARAMETERS = {
    "min_area": {
        "type": "int",
        "min": 0, "max": 1000000,
        "default": 100,
        "step": 50
    },
    "max_area": {
        "type": "int",
        "min": 0, "max": 10000000,
        "default": 500000,
        "step": 1000
    },
    "min_circularity": {
        "type": "float",
        "min": 0.0, "max": 1.0,
        "default": 0.0,
        "step": 0.05,
        "decimals": 2
    },
    "draw_contour": {
        "type": "bool",
        "default": True
    },
    "draw_bounding_box": {
        "type": "bool",
        "default": True
    },
    "draw_centroid": {
        "type": "bool",
        "default": True
    },
    "show_stats_badge": {
        "type": "bool",
        "default": True
    },
    "pass_fail_check": {
        "type": "bool",
        "default": False
    },
    "target_count_min": {
        "type": "int",
        "min": 0, "max": 1000,
        "default": 1
    },
    "target_count_max": {
        "type": "int",
        "min": 0, "max": 1000,
        "default": 100
    }
}

def process(frame, params=None):
    """
    Industrial Contour & Blob Analysis with Area/Circularity Filtering and Pass/Fail HUD.
    """
    if frame is None:
        return None
    if params is None:
        params = {}

    min_area = int(params.get("min_area", 100))
    max_area = int(params.get("max_area", 500000))
    min_circ = float(params.get("min_circularity", 0.0))
    draw_cnt = bool(params.get("draw_contour", True))
    draw_bbox = bool(params.get("draw_bounding_box", True))
    draw_centroid = bool(params.get("draw_centroid", True))
    show_badge = bool(params.get("show_stats_badge", True))
    pass_fail = bool(params.get("pass_fail_check", False))
    target_min = int(params.get("target_count_min", 1))
    target_max = int(params.get("target_count_max", 100))

    # Convert to grayscale binary
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Determine if already binary
    unique_vals = np.unique(gray)
    if len(unique_vals) <= 4 and (0 in unique_vals or 255 in unique_vals):
        binary = gray
    else:
        # Auto Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame.copy()
    valid_blobs = 0
    total_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-6)
        if min_circ > 0.0 and circularity < min_circ:
            continue

        valid_blobs += 1
        total_area += area

        if draw_cnt:
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2, cv2.LINE_AA)

        if draw_bbox:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 180, 0), 2, cv2.LINE_AA)

        if draw_centroid:
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # Heads-Up Display Badge
    if show_badge:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1

        is_pass = (target_min <= valid_blobs <= target_max) if pass_fail else None
        
        if pass_fail:
            status_text = "PASS" if is_pass else "FAIL"
            status_color = (0, 180, 0) if is_pass else (0, 0, 220)
            text_lines = [
                f"Status: {status_text} (Target: {target_min}-{target_max})",
                f"Objects: {valid_blobs}",
                f"Total Area: {int(total_area)} px"
            ]
        else:
            status_color = (40, 40, 40)
            text_lines = [
                f"Detected Objects: {valid_blobs}",
                f"Total Area: {int(total_area)} px"
            ]

        # Calculate badge dimensions
        line_height = 20
        badge_w = 260
        badge_h = len(text_lines) * line_height + 14

        # Draw semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + badge_w, 10 + badge_h), status_color, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + badge_w, 10 + badge_h), (255, 255, 255), 1, cv2.LINE_AA)

        # Draw text lines
        for i, line in enumerate(text_lines):
            y_pos = 10 + (i + 1) * line_height
            cv2.putText(annotated, line, (18, y_pos), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return annotated
