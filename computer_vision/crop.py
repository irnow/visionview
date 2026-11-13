import cv2

PARAMETERS = {
    "mode": {
        "type": "enum", 
        "options": ["Input", "Drag"], 
        "default": "Input"
    },
    "width": {
        "type": "int",
        "min": 0,
        "max": 4000,
        "default": 640
    },
    "height": {
        "type": "int",
        "min": 0,
        "max": 4000,
        "default": 480
    },
    "x_pos": {
        "type": "int",
        "min": 0,
        "max": 4000,
        "default": 0
    },
    "y_pos": {
        "type": "int",
        "min": 0,
        "max": 4000,
        "default": 0
    }
}

def process(frame, params=None):
    """
    Cropping Tool: Supports manual numeric Input and interactive Drag modes.
    Ensures output is always valid BGR or Grayscale.
    """
    if frame is None:
        return None

    if params is None:
        params = {}

    h, w = frame.shape[:2]
    
    mode = params.get("mode", PARAMETERS["mode"]["default"])
    raw_w = int(params.get("width", PARAMETERS["width"]["default"]))
    raw_h = int(params.get("height", PARAMETERS["height"]["default"]))

    if mode == "Drag" and (raw_w <= 0 or raw_h <= 0):
        # Reset mode in Drag: display full uncropped frame so user can select an ROI
        return frame

    cw = max(1, min(raw_w, w))
    ch = max(1, min(raw_h, h))

    if mode in ("Input", "Drag", "dynamic"):
        start_x = int(params.get("x_pos", PARAMETERS["x_pos"]["default"]))
        start_y = int(params.get("y_pos", PARAMETERS["y_pos"]["default"]))

        max_x = max(0, w - cw)
        max_y = max(0, h - ch)
        start_x = max(0, min(start_x, max_x))
        start_y = max(0, min(start_y, max_y))
    elif mode == "center":
        start_x = max(0, (w - cw) // 2)
        start_y = max(0, (h - ch) // 2)
    else:
        start_x = int(params.get("x_pos", 0))
        start_y = int(params.get("y_pos", 0))
        max_x = max(0, w - cw)
        max_y = max(0, h - ch)
        start_x = max(0, min(start_x, max_x))
        start_y = max(0, min(start_y, max_y))

    end_x = start_x + cw
    end_y = start_y + ch
    
    cropped = frame[start_y:end_y, start_x:end_x]

    if len(cropped.shape) == 2:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    return cropped