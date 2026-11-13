import cv2
import numpy as np
import os

_TEMPLATE_CACHE = {}
_LIVE_TEMPLATE = None
_LIVE_TEMPLATE_PATH = None

PARAMETERS = {
    "source": {
        "type": "enum",
        "options": ["File", "Drag"],
        "default": "File"
    },
    "template_file": {
        "type": "file",
        "default": "",
        "filter": "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*.*)",
        "allow_live_capture": True
    },
    "match_method": {
        "type": "enum",
        "options": ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
        "default": "TM_CCOEFF_NORMED"
    },
    "threshold": {
        "type": "float",
        "min": 0.1, "max": 1.0,
        "default": 0.65,
        "step": 0.05,
        "decimals": 2
    },
    "mode": {
        "type": "enum",
        "options": ["Multi-Match", "Best Match"],
        "default": "Multi-Match"
    },
    "max_matches": {
        "type": "int",
        "min": 1, "max": 100,
        "default": 20
    },
    "draw_crosshair": {
        "type": "bool",
        "default": True
    }
}

_METHOD_MAP = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
}

def set_live_template(template_bgr, save_path=None):
    """
    Sets the active template cropped directly from live video.
    """
    global _LIVE_TEMPLATE, _LIVE_TEMPLATE_PATH
    if template_bgr is not None:
        _LIVE_TEMPLATE = template_bgr.copy()
        _LIVE_TEMPLATE_PATH = save_path
        if save_path:
            _TEMPLATE_CACHE[save_path] = _LIVE_TEMPLATE

def get_live_template():
    global _LIVE_TEMPLATE
    return _LIVE_TEMPLATE

def clear_live_template():
    global _LIVE_TEMPLATE, _LIVE_TEMPLATE_PATH
    _LIVE_TEMPLATE = None
    _LIVE_TEMPLATE_PATH = None

def get_template(template_path):
    global _LIVE_TEMPLATE, _LIVE_TEMPLATE_PATH
    # 1. Check live in-memory template
    if _LIVE_TEMPLATE is not None:
        if not template_path or template_path == _LIVE_TEMPLATE_PATH or template_path == "live_template":
            return _LIVE_TEMPLATE

    # 2. Check disk file
    if not template_path or not os.path.isfile(template_path):
        return _LIVE_TEMPLATE

    cached = _TEMPLATE_CACHE.get(template_path)
    if cached is None:
        tpl = cv2.imread(template_path)
        if tpl is not None:
            _TEMPLATE_CACHE[template_path] = tpl
            cached = tpl
    return cached

def process(frame, params=None):
    """
    Industrial Template Matching / Pattern Alignment Module.
    Supports templates loaded from disk or highlighted directly on the live video module.
    """
    if frame is None:
        return None
    if params is None:
        params = {}

    source = params.get("source", "File")
    template_file = params.get("template_file", "")
    method_name = params.get("match_method", "TM_CCOEFF_NORMED")
    threshold = float(params.get("threshold", 0.75))
    mode = params.get("mode", "Best Match")
    max_matches = int(params.get("max_matches", 5))
    draw_crosshair = bool(params.get("draw_crosshair", True))

    template = None
    if source == "Drag" and _LIVE_TEMPLATE is not None:
        template = _LIVE_TEMPLATE
    if template is None and template_file:
        template = get_template(template_file)
    if template is None:
        template = get_live_template()

    if template is None:
        annotated = frame.copy()
        msg = ("Template Matching [Drag]: Click 'Highlight on Video' in parameters to crop template"
               if source == "Drag" else
               "Template Matching [File]: Select a template image file in parameters")
        cv2.putText(
            annotated, msg,
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1, cv2.LINE_AA
        )
        return annotated

    th, tw = template.shape[:2]
    fh, fw = frame.shape[:2]

    if th > fh or tw > fw:
        annotated = frame.copy()
        cv2.putText(
            annotated, "Template Matching: Template is larger than input frame!",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA
        )
        return annotated

    # Convert to grayscale for robust normalized matching
    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    cv_method = _METHOD_MAP.get(method_name, cv2.TM_CCOEFF_NORMED)
    res = cv2.matchTemplate(f_gray, t_gray, cv_method)

    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == "Best Match":
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if cv_method == cv2.TM_SQDIFF_NORMED:
            score = 1.0 - float(min_val)
            top_left = min_loc
        else:
            score = float(max_val)
            top_left = max_loc

        bx, by = top_left
        is_match = score >= threshold

        color = (0, 220, 0) if is_match else (0, 0, 220)
        tag = f"Match: {score*100:.1f}%" if is_match else f"No Match: {score*100:.1f}% (Thresh {threshold*100:.0f}%)"

        cv2.rectangle(annotated, (bx, by), (bx + tw, by + th), color, 2, cv2.LINE_AA)

        if draw_crosshair and is_match:
            cx, cy = bx + tw // 2, by + th // 2
            cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), color, 1, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), color, 1, cv2.LINE_AA)

        # Draw badge
        (tw_text, th_text), _ = cv2.getTextSize(tag, font, 0.5, 1)
        ty = max(0, by - th_text - 6)
        cv2.rectangle(annotated, (bx, ty), (bx + tw_text + 8, ty + th_text + 6), color, -1)
        cv2.putText(annotated, tag, (bx + 4, ty + th_text + 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Guidance hint for discovering all patterns
        cv2.putText(annotated, "Best Match: 1 pattern found (Select 'Multi-Match' in parameters to find all patterns)", 
                    (15, 30), font, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    else: # Multi-Match
        if cv_method == cv2.TM_SQDIFF_NORMED:
            match_mask = (res <= (1.0 - threshold))
            scores_matrix = 1.0 - res
        else:
            match_mask = (res >= threshold)
            scores_matrix = res

        loc_y, loc_x = np.where(match_mask)
        boxes = []
        scores = []
        for y, x in zip(loc_y, loc_x):
            boxes.append([int(x), int(y), int(tw), int(th)])
            scores.append(float(scores_matrix[y, x]))

        # Non-Maximum Suppression to eliminate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=threshold, nms_threshold=0.3)
        if len(indices) > 0:
            indices = indices.flatten()[:max_matches]
            for idx in indices:
                bx, by, bw, bh = boxes[idx]
                sc = scores[idx]

                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 220, 0), 2, cv2.LINE_AA)
                if draw_crosshair:
                    cx, cy = bx + bw // 2, by + bh // 2
                    cv2.line(annotated, (cx - 8, cy), (cx + 8, cy), (0, 220, 0), 1, cv2.LINE_AA)
                    cv2.line(annotated, (cx, cy - 8), (cx, cy + 8), (0, 220, 0), 1, cv2.LINE_AA)

                tag = f"{sc*100:.1f}%"
                cv2.putText(annotated, tag, (bx + 3, by - 4), font, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

            count_tag = f"Patterns Found: {len(indices)}"
            cv2.putText(annotated, count_tag, (15, 30), font, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "Multi-Match: No patterns above threshold (try lowering threshold)", (15, 30), font, 0.55, (0, 0, 220), 1, cv2.LINE_AA)

    return annotated
