import cv2
import numpy as np
import os
import sys

# Optional Tesseract support
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

# Optional Pillow support for high-fidelity industrial TrueType glyph templates
try:
    from PIL import Image, ImageFont, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

PARAMETERS = {
    "engine": {
        "type": "enum",
        "options": ["Industrial OCR (OpenCV)", "Seven-Segment Display", "Tesseract (Optional)"],
        "default": "Industrial OCR (OpenCV)"
    },
    "polarity": {
        "type": "enum",
        "options": ["Auto (Laser / Dark Markings)", "Light on Dark", "Dark on Light"],
        "default": "Auto (Laser / Dark Markings)"
    },
    "char_type": {
        "type": "enum",
        "options": ["Alphanumeric", "Digits Only", "Uppercase Only"],
        "default": "Alphanumeric"
    },
    "invert_binary": {
        "type": "bool",
        "default": False
    },
    "contrast_boost": {
        "type": "bool",
        "default": True
    },
    "draw_bounding_box": {
        "type": "bool",
        "default": True
    },
    "draw_text": {
        "type": "bool",
        "default": True
    },
    "min_confidence": {
        "type": "float",
        "min": 0.1, "max": 1.0,
        "default": 0.35,
        "step": 0.05,
        "decimals": 2
    }
}

# ---------------------------------------------------------------------------
# 1. Industrial Glyph Engine (Zero external dependencies, pure OpenCV + PIL)
# ---------------------------------------------------------------------------
_GLYPH_CACHE = {}

def _build_glyph_cache():
    """Generates normalized 24x32 binary templates for standard industrial fonts"""
    global _GLYPH_CACHE
    if _GLYPH_CACHE:
        return _GLYPH_CACHE

    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cache = {ch: [] for ch in chars}

    # 1. TrueType industrial fonts (Arial, Consolas, Segoe UI, Calibri)
    if HAS_PIL:
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/cour.ttf"
        ]
        loaded_fonts = []
        for fp in font_candidates:
            if os.path.exists(fp):
                for sz in [32, 38, 44]:
                    try:
                        loaded_fonts.append(ImageFont.truetype(fp, sz))
                    except Exception:
                        pass

        for ch in chars:
            for f in loaded_fonts:
                im = Image.new("L", (70, 70), color=0)
                draw = ImageDraw.Draw(im)
                draw.text((10, 5), ch, fill=255, font=f)
                arr = np.array(im)
                pts = cv2.findNonZero(arr)
                if pts is not None:
                    bx, by, bw, bh = cv2.boundingRect(pts)
                    if bw > 3 and bh > 5:
                        patch = arr[by:by+bh, bx:bx+bw]
                        resized = cv2.resize(patch, (24, 32), interpolation=cv2.INTER_AREA)
                        cache[ch].append(resized)

    # 2. Built-in Hershey vector fonts fallback
    hershey_fonts = [
        (cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2),
        (cv2.FONT_HERSHEY_DUPLEX, 1.2, 2),
    ]
    for ch in chars:
        for font, scale, thick in hershey_fonts:
            canvas = np.zeros((80, 80), dtype=np.uint8)
            cv2.putText(canvas, ch, (15, 60), font, scale, 255, thick, cv2.LINE_AA)
            pts = cv2.findNonZero(canvas)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                if bw > 2 and bh > 2:
                    crop = canvas[by:by+bh, bx:bx+bw]
                    resized = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_AREA)
                    cache[ch].append(resized)

    # Clean empty
    _GLYPH_CACHE = {k: v for k, v in cache.items() if len(v) > 0}
    return _GLYPH_CACHE


def process_industrial_ocr(frame, char_type="Alphanumeric", min_conf=0.35,
                           polarity="Auto (Laser / Dark Markings)",
                           contrast_boost=True, invert_bin=False,
                           draw_box=True, draw_text=True):
    """
    High-speed, pure OpenCV industrial character reader.
    Extracts text markings (chip labels, serials, part numbers) without external binaries.
    Uses Black-Hat morphology, internal contour extraction, line grouping, and multi-font glyph matching.
    """
    annotated = frame.copy()
    h_img, w_img = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Grayscale & Contrast Boost
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    if contrast_boost:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
    else:
        gray_enhanced = gray

    # 2. Morphological Isolation (Black-Hat for dark laser marks, Top-Hat for light ink)
    ksize = max(11, min(31, int(min(h_img, w_img) * 0.05)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    blackhat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, kernel)
    bh_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh_bh = cv2.threshold(bh_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    tophat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_TOPHAT, kernel)
    th_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh_th = cv2.threshold(th_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if invert_bin:
        thresh = thresh_th if "Dark" in polarity else thresh_bh
    else:
        thresh = thresh_bh if "Light on Dark" not in polarity else thresh_th

    # 3. Candidate Contour Extraction (RETR_LIST captures internal characters inside IC bodies)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h + 1e-5)
        # Industrial characters have reasonable height and aspect ratio
        if 14 <= h <= int(h_img * 0.6) and 6 <= w <= int(w_img * 0.4) and 0.18 <= aspect <= 1.25 and (w * h) >= 36:
            raw_boxes.append([x, y, w, h])

    if not raw_boxes:
        return annotated

    # 4. Non-Maximum Suppression (eliminate overlapping duplicate boxes)
    scores = [float(b[3]) for b in raw_boxes]
    indices = cv2.dnn.NMSBoxes(raw_boxes, scores, score_threshold=0.0, nms_threshold=0.3)
    boxes = [raw_boxes[i] for i in indices]

    # 5. Remove nested/contained boxes (e.g. inner holes of 0, O, A, P, D, B)
    non_contained = []
    for i, b1 in enumerate(boxes):
        x1, y1, w1, h1 = b1
        contained = False
        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            x2, y2, w2, h2 = b2
            if x1 >= x2 - 2 and y1 >= y2 - 2 and (x1 + w1) <= (x2 + w2 + 2) and (y1 + h1) <= (y2 + h2 + 2):
                contained = True
                break
        if not contained:
            non_contained.append(b1)
    boxes = non_contained

    if not boxes:
        return annotated

    # 6. Group into text lines (similar vertical coordinate)
    boxes.sort(key=lambda b: b[1])
    lines = []
    for b in boxes:
        placed = False
        for line in lines:
            if abs(line[0][1] - b[1]) <= 14:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    # 7. Cluster words in each line based on character pitch/spacing
    words = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        curr_w = [line[0]]
        for i in range(1, len(line)):
            prev = line[i-1]
            curr = line[i]
            gap = curr[0] - (prev[0] + prev[2])
            # Characters belonging to the same word are closely spaced
            if gap <= 18:
                curr_w.append(curr)
            else:
                if len(curr_w) >= 2:
                    words.append(curr_w)
                curr_w = [curr]
        if len(curr_w) >= 2:
            words.append(curr_w)

    # 8. Glyph matching for each character in detected words
    glyph_cache = _build_glyph_cache()
    if char_type == "Digits Only":
        allowed_chars = {k: v for k, v in glyph_cache.items() if k in "0123456789"}
    elif char_type == "Uppercase Only":
        allowed_chars = {k: v for k, v in glyph_cache.items() if k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    else:
        allowed_chars = glyph_cache

    all_recognized_words = []

    for word_boxes in words:
        med_h = int(np.median([b[3] for b in word_boxes]))
        med_y = int(np.median([b[1] for b in word_boxes]))
        word_chars = []

        for (x, y, w, h) in word_boxes:
            # Normalize crop coordinates to full line height/baseline
            use_y = max(0, min(med_y, y))
            use_h = max(h, med_h)
            if use_y + use_h > h_img:
                use_h = h_img - use_y

            char_roi = gray_enhanced[use_y:use_y+use_h, x:x+w]
            clahe_local = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enh = clahe_local.apply(char_roi)
            _, bin_inv = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            norm_char = cv2.resize(bin_inv, (24, 32), interpolation=cv2.INTER_AREA)

            best_ch, best_sc = "?", -1.0
            for ch, tpl_list in allowed_chars.items():
                for tpl in tpl_list:
                    res = cv2.matchTemplate(norm_char, tpl, cv2.TM_CCOEFF_NORMED)
                    sc = float(res[0][0])
                    if sc > best_sc:
                        best_sc = sc
                        best_ch = ch

            if best_sc >= min_conf:
                word_chars.append((x, use_y, w, use_h, best_ch, best_sc))

        if word_chars:
            w_text = "".join(c[4] for c in word_chars)
            all_recognized_words.append(w_text)

            for (x, y, w, h, ch, sc) in word_chars:
                if draw_box:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2, cv2.LINE_AA)
                if draw_text:
                    tag = f"{ch} ({int(sc*100)}%)"
                    (tw, th), _ = cv2.getTextSize(tag, font, 0.45, 1)
                    ty = max(0, y - 4)
                    cv2.rectangle(annotated, (x, ty - th - 2), (x + tw + 4, ty + 2), (0, 180, 0), -1)
                    cv2.putText(annotated, tag, (x + 2, ty), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 9. Top Summary Banner
    if all_recognized_words:
        summary_text = "OCR: " + " ".join(all_recognized_words)
        (bw, bh), _ = cv2.getTextSize(summary_text, font, 0.6, 2)
        cv2.rectangle(annotated, (10, 10), (10 + bw + 14, 10 + bh + 14), (0, 140, 0), -1)
        cv2.putText(annotated, summary_text, (16, 10 + bh + 4), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# 2. Seven-Segment Display Engine (Built-in, Zero Dependencies)
# ---------------------------------------------------------------------------
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): "0",
    (0, 0, 1, 0, 0, 1, 0): "1",
    (1, 0, 1, 1, 1, 0, 1): "2",
    (1, 0, 1, 1, 0, 1, 1): "3",
    (0, 1, 1, 1, 0, 1, 0): "4",
    (1, 1, 0, 1, 0, 1, 1): "5",
    (1, 1, 0, 1, 1, 1, 1): "6",
    (1, 0, 1, 0, 0, 1, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9"
}

def decode_seven_segment(roi):
    h, w = roi.shape[:2]
    if h < 14 or w < 8:
        return None

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    segments = [
        ((int(w * 0.2), 0), (int(w * 0.8), int(h * 0.15))),
        ((0, int(h * 0.15)), (int(w * 0.3), int(h * 0.5))),
        ((int(w * 0.7), int(h * 0.15)), (w, int(h * 0.5))),
        ((int(w * 0.2), int(h * 0.42)), (int(w * 0.8), int(h * 0.58))),
        ((0, int(h * 0.5)), (int(w * 0.3), int(h * 0.85))),
        ((int(w * 0.7), int(h * 0.5)), (w, int(h * 0.85))),
        ((int(w * 0.2), int(h * 0.85)), (int(w * 0.8), h))
    ]

    on_segments = []
    for ((x1, y1), (x2, y2)) in segments:
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            on_segments.append(0)
            continue
        seg_roi = thresh[y1:y2, x1:x2]
        total = cv2.countNonZero(seg_roi)
        area = (x2 - x1) * (y2 - y1)
        on_segments.append(1 if (total / float(area)) > 0.35 else 0)

    key = tuple(on_segments)
    return DIGITS_LOOKUP.get(key, None)

def process_seven_segment_engine(frame, draw_box=True, draw_text=True):
    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX

    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / float(w + 1e-5)
        if 1.2 <= aspect <= 3.2 and 18 <= h <= 300 and w >= 8:
            digit_boxes.append((x, y, w, h))

    digit_boxes.sort(key=lambda b: b[0])
    recognized = []
    for (x, y, w, h) in digit_boxes:
        roi = frame[y:y+h, x:x+w]
        digit = decode_seven_segment(roi)
        if digit is not None:
            recognized.append((x, y, w, h, digit))
            if draw_box:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            if draw_text:
                cv2.putText(annotated, digit, (x + 2, y - 5), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if recognized:
        full_val = "".join([d[4] for d in recognized])
        cv2.putText(annotated, f"Val: {full_val}", (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# 3. Main Node Entry Point
# ---------------------------------------------------------------------------
def process(frame, params=None):
    """
    Optical Character Recognition (OCR) Module.
    Defaults to Industrial OCR (OpenCV) with zero external setup.
    """
    if frame is None:
        return None
    if params is None:
        params = {}

    engine = params.get("engine", "Industrial OCR (OpenCV)")
    char_type = params.get("char_type", "Alphanumeric")
    polarity = params.get("polarity", "Auto (Laser / Dark Markings)")
    invert_bin = bool(params.get("invert_binary", False))
    contrast_boost = bool(params.get("contrast_boost", True))
    draw_box = bool(params.get("draw_bounding_box", True))
    draw_text = bool(params.get("draw_text", True))
    min_conf = float(params.get("min_confidence", 0.35))

    # 1. Industrial OCR (OpenCV) - Default
    if engine in ("Industrial OCR (OpenCV)", "Auto"):
        return process_industrial_ocr(
            frame, char_type=char_type, min_conf=min_conf,
            polarity=polarity,
            contrast_boost=contrast_boost, invert_bin=invert_bin,
            draw_box=draw_box, draw_text=draw_text
        )

    # 2. Seven-Segment Display Engine
    if engine == "Seven-Segment Display":
        return process_seven_segment_engine(frame, draw_box=draw_box, draw_text=draw_text)

    # 3. Optional Tesseract Engine
    if engine == "Tesseract (Optional)":
        if HAS_PYTESSERACT:
            try:
                pytesseract.get_tesseract_version()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if contrast_boost:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                if invert_bin:
                    gray = cv2.bitwise_not(gray)

                cfg = "--psm 6"
                if char_type == "Digits Only":
                    cfg += " -c tessedit_char_whitelist=0123456789.-"
                elif char_type == "Uppercase Only":
                    cfg += " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"

                data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
                annotated = frame.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                n_boxes = len(data.get("text", []))
                for i in range(n_boxes):
                    text = data["text"][i].strip()
                    conf = float(data["conf"][i]) if "conf" in data else -1.0
                    if text and conf >= (min_conf * 100.0):
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        if draw_box:
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2, cv2.LINE_AA)
                        if draw_text:
                            lbl = f"{text} ({int(conf)}%)"
                            (tw, th), _ = cv2.getTextSize(lbl, font, 0.45, 1)
                            ty = max(0, y - th - 4)
                            cv2.rectangle(annotated, (x, ty), (x + tw + 4, ty + th + 4), (0, 180, 0), -1)
                            cv2.putText(annotated, lbl, (x + 2, ty + th + 1), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                return annotated
            except Exception as e:
                print(f"Tesseract inference error: {e}")

        annotated = frame.copy()
        cv2.putText(
            annotated, "OCR: Tesseract binary not found. Switched to Industrial OCR (OpenCV)",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1, cv2.LINE_AA
        )
        return process_industrial_ocr(
            frame, char_type=char_type, min_conf=min_conf,
            polarity=polarity,
            contrast_boost=contrast_boost, invert_bin=invert_bin,
            draw_box=draw_box, draw_text=draw_text
        )

    return frame
