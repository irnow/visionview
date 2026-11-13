import cv2

PARAMETERS = {
    "operation": {
        "type": "enum",
        "options": ["Erode", "Dilate", "Open", "Close", "Gradient", "Top Hat", "Black Hat"],
        "default": "Close"
    },
    "kernel_shape": {
        "type": "enum",
        "options": ["Rectangle", "Ellipse", "Cross"],
        "default": "Rectangle"
    },
    "kernel_size": {
        "type": "int",
        "min": 1, "max": 51,
        "default": 3,
        "step": 2
    },
    "iterations": {
        "type": "int",
        "min": 1, "max": 10,
        "default": 1,
        "step": 1
    }
}

_SHAPE_MAP = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS
}

_OP_MAP = {
    "Erode": cv2.MORPH_ERODE,
    "Dilate": cv2.MORPH_DILATE,
    "Open": cv2.MORPH_OPEN,
    "Close": cv2.MORPH_CLOSE,
    "Gradient": cv2.MORPH_GRADIENT,
    "Top Hat": cv2.MORPH_TOPHAT,
    "Black Hat": cv2.MORPH_BLACKHAT
}

def process(frame, params=None):
    """
    Industrial Morphological Filtering Module (Erode, Dilate, Open, Close, etc.).
    """
    if frame is None:
        return None
    if params is None:
        params = {}

    op_name = params.get("operation", "Close")
    shape_name = params.get("kernel_shape", "Rectangle")
    k_size = int(params.get("kernel_size", 3))
    iters = max(1, int(params.get("iterations", 1)))

    # Ensure kernel size is positive and odd
    if k_size % 2 == 0:
        k_size += 1
    k_size = max(1, k_size)

    morph_shape = _SHAPE_MAP.get(shape_name, cv2.MORPH_RECT)
    morph_op = _OP_MAP.get(op_name, cv2.MORPH_CLOSE)

    kernel = cv2.getStructuringElement(morph_shape, (k_size, k_size))

    if op_name == "Erode":
        return cv2.erode(frame, kernel, iterations=iters)
    elif op_name == "Dilate":
        return cv2.dilate(frame, kernel, iterations=iters)
    else:
        return cv2.morphologyEx(frame, morph_op, kernel, iterations=iters)
