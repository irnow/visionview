# ==============================================================================
# VisionView - Split View Compositor (Raw & Processed View Comparison)
# ==============================================================================

import cv2
import numpy as np


def ensure_bgr(img):
    """Ensures input image is a 3-channel BGR numpy array."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def draw_view_badge(img, text, dot_color, x, y):
    """
    Renders a semi-transparent dark status badge with a colored indicator dot
    and antialiased text (e.g. '[● RAW]' or '[● PROCESSED]').
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = tw + 26
    box_h = th + 14

    h, w = img.shape[:2]
    x = max(4, min(w - box_w - 4, x))
    y = max(4, min(h - box_h - 4, y))

    # Translucent dark background overlay
    sub = img[y:y + box_h, x:x + box_w]
    if sub.shape[0] == box_h and sub.shape[1] == box_w:
        dark_overlay = np.full((box_h, box_w, 3), 20, dtype=np.uint8)
        blended = cv2.addWeighted(dark_overlay, 0.72, sub, 0.28, 0)
        img[y:y + box_h, x:x + box_w] = blended

    # Border
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (70, 70, 70), 1, cv2.LINE_AA)

    # Indicator Dot
    dot_x = x + 10
    dot_y = y + box_h // 2
    cv2.circle(img, (dot_x, dot_y), 4, dot_color, -1, cv2.LINE_AA)

    # Badge text
    cv2.putText(img, text, (x + 19, y + th + 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def compose_split_view(raw_img, processed_img, mode="side_by_side", split_ratio=0.5, pipeline_enabled=True):
    """
    Composes a side-by-side or curtain split view comparing the raw input feed
    with the processed computer vision pipeline output.

    Parameters:
        raw_img: Original unprocessed camera or static image frame
        processed_img: Output frame from computer vision pipeline
        mode: 'side_by_side' (default), 'curtain', or 'top_bottom'
        split_ratio: Fraction of view split for 'curtain' mode (0.1 to 0.9)
        pipeline_enabled: Whether CV pipeline is active (updates badge label)

    Returns:
        Composed 3-channel BGR numpy array with badges and divider lines
    """
    if raw_img is None and processed_img is None:
        return None
    if raw_img is None:
        return ensure_bgr(processed_img)
    if processed_img is None:
        return ensure_bgr(raw_img)

    raw_bgr = ensure_bgr(raw_img)
    proc_bgr = ensure_bgr(processed_img)

    rh, rw = raw_bgr.shape[:2]
    ph, pw = proc_bgr.shape[:2]

    # Normalize dimensions if processed size changed (e.g. from resize/crop filter)
    if (ph, pw) != (rh, rw):
        proc_bgr = cv2.resize(proc_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        ph, pw = rh, rw

    proc_label = "PROCESSED" if pipeline_enabled else "PROCESSED (BYPASS)"
    proc_dot = (237, 162, 0) if pipeline_enabled else (128, 128, 128) # Cyan/blue if active, grey if bypassed
    raw_dot = (50, 205, 50) # Green for live raw stream

    # -------------------------------------------------------------
    # 1. SIDE-BY-SIDE (Default Horizontal Split)
    # -------------------------------------------------------------
    if mode == "side_by_side":
        # Make a copy to prevent in-place mutation of caller's raw frame
        raw_side = raw_bgr.copy()
        proc_side = proc_bgr.copy()

        composite = np.hstack([raw_side, proc_side])
        total_w = composite.shape[1]
        div_x = rw

        # Divider line: Subtle shadow with vibrant cyan center line
        cv2.line(composite, (div_x - 1, 0), (div_x - 1, rh), (20, 20, 20), 1)
        cv2.line(composite, (div_x, 0), (div_x, rh), (237, 162, 0), 2)
        cv2.line(composite, (div_x + 1, 0), (div_x + 1, rh), (20, 20, 20), 1)

        # Badges
        draw_view_badge(composite, "RAW", raw_dot, 12, 12)
        draw_view_badge(composite, proc_label, proc_dot, div_x + 12, 12)
        return composite

    # -------------------------------------------------------------
    # 2. CURTAIN / SPLIT SCREEN (50/50 Swipe Comparison)
    # -------------------------------------------------------------
    elif mode in ("curtain", "split_screen"):
        composite = np.empty_like(raw_bgr)
        split_ratio = max(0.05, min(0.95, split_ratio))
        split_x = int(rw * split_ratio)

        # Left is Raw, Right is Processed
        composite[:, :split_x] = raw_bgr[:, :split_x]
        composite[:, split_x:] = proc_bgr[:, split_x:]

        # Divider Line
        cv2.line(composite, (split_x - 1, 0), (split_x - 1, rh), (20, 20, 20), 1)
        cv2.line(composite, (split_x, 0), (split_x, rh), (255, 255, 255), 2)
        cv2.line(composite, (split_x + 1, 0), (split_x + 1, rh), (20, 20, 20), 1)

        # Center handle / grip icon
        cy = rh // 2
        cv2.circle(composite, (split_x, cy), 10, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(composite, (split_x, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(composite, (split_x - 4, cy), (split_x - 2, cy), (255, 255, 255), 1)
        cv2.line(composite, (split_x + 2, cy), (split_x + 4, cy), (255, 255, 255), 1)

        # Badges
        draw_view_badge(composite, "RAW", raw_dot, 12, 12)
        draw_view_badge(composite, proc_label, proc_dot, rw - 130, 12)
        return composite

    # -------------------------------------------------------------
    # 3. TOP-BOTTOM (Vertical Stack Split)
    # -------------------------------------------------------------
    elif mode == "top_bottom":
        raw_side = raw_bgr.copy()
        proc_side = proc_bgr.copy()

        composite = np.vstack([raw_side, proc_side])
        div_y = rh

        cv2.line(composite, (0, div_y - 1), (rw, div_y - 1), (20, 20, 20), 1)
        cv2.line(composite, (0, div_y), (rw, div_y), (237, 162, 0), 2)
        cv2.line(composite, (0, div_y + 1), (rw, div_y + 1), (20, 20, 20), 1)

        draw_view_badge(composite, "RAW", raw_dot, 12, 12)
        draw_view_badge(composite, proc_label, proc_dot, 12, div_y + 12)
        return composite

    else:
        # Fallback to side by side
        return compose_split_view(raw_bgr, proc_bgr, mode="side_by_side", split_ratio=split_ratio, pipeline_enabled=pipeline_enabled)
