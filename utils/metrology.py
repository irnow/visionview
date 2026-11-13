import math
import numpy as np
import cv2


def fit_circle_3p(p1, p2, p3):
    """
    Fits a circle passing through 3 non-collinear 2D points.
    Returns: (center_x, center_y, radius) or None if points are collinear or invalid.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])

    # Determinant for collinearity check
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-7:
        return None

    sq1 = x1**2 + y1**2
    sq2 = x2**2 + y2**2
    sq3 = x3**2 + y3**2

    cx = (sq1 * (y2 - y3) + sq2 * (y3 - y1) + sq3 * (y1 - y2)) / d
    cy = (sq1 * (x3 - x2) + sq2 * (x1 - x3) + sq3 * (x2 - x1)) / d
    radius = math.hypot(x1 - cx, y1 - cy)

    if math.isnan(cx) or math.isnan(cy) or math.isnan(radius) or radius < 1e-4:
        return None

    return (float(cx), float(cy), float(radius))


def compute_angle_deg(p1, vertex, p2):
    """
    Computes angle in degrees [0.0, 180.0] between ray vertex->p1 and ray vertex->p2.
    """
    vx1 = float(p1[0]) - float(vertex[0])
    vy1 = float(p1[1]) - float(vertex[1])
    vx2 = float(p2[0]) - float(vertex[0])
    vy2 = float(p2[1]) - float(vertex[1])

    len1 = math.hypot(vx1, vy1)
    len2 = math.hypot(vx2, vy2)

    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0

    dot = (vx1 * vx2) + (vy1 * vy2)
    cos_theta = dot / (len1 * len2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    rad = math.acos(cos_theta)
    return math.degrees(rad)


def run_1d_edge_caliper(image, p1, p2, stripe_width=5, min_gradient_rel=0.18):
    """
    Samples a 1D intensity profile along the line from p1 to p2 with perpendicular stripe averaging,
    and performs sub-pixel edge detection using parabolic peak interpolation.

    Returns dict with detected edges and sub-pixel distance.
    """
    if image is None or image.size == 0:
        return {'success': False, 'error': 'Empty image', 'edges': [], 'distance': 0.0}

    # Convert to single-channel grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    line_len = math.hypot(x2 - x1, y2 - y1)
    if line_len < 4.0:
        return {'success': False, 'error': 'Line too short', 'edges': [], 'distance': 0.0}

    num_samples = max(int(round(line_len)), 10)
    t_vals = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

    # Unit vector along line and normal vector
    ux = (x2 - x1) / line_len
    uy = (y2 - y1) / line_len
    nx = -uy
    ny = ux

    # Center points along the line
    xs = x1 + t_vals * (x2 - x1)
    ys = y1 + t_vals * (y2 - y1)

    half_w = max(0, stripe_width // 2)
    offsets = list(range(-half_w, half_w + 1)) if half_w > 0 else [0]
    
    h_img, w_img = gray.shape[:2]
    accum_profile = np.zeros(num_samples, dtype=np.float32)

    for off in offsets:
        cx = np.clip(np.round(xs + off * nx).astype(np.int32), 0, w_img - 1)
        cy = np.clip(np.round(ys + off * ny).astype(np.int32), 0, h_img - 1)
        accum_profile += gray[cy, cx].astype(np.float32)

    profile = accum_profile / float(len(offsets))

    # 1D Gradient calculation
    grad = np.gradient(profile)
    grad_mag = np.abs(grad)

    # Smooth gradient magnitude with a small 5-tap Gaussian-like binomial kernel
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    smoothed_mag = np.convolve(grad_mag, kernel, mode='same')

    max_mag = float(np.max(smoothed_mag))
    threshold = max(3.0, max_mag * float(min_gradient_rel))

    detected_edges = []
    for i in range(1, num_samples - 1):
        if smoothed_mag[i] >= threshold and smoothed_mag[i] >= smoothed_mag[i - 1] and smoothed_mag[i] > smoothed_mag[i + 1]:
            # 3-Point parabolic interpolation for sub-pixel peak location
            m0 = float(smoothed_mag[i - 1])
            m1 = float(smoothed_mag[i])
            m2 = float(smoothed_mag[i + 1])
            denom = 2.0 * (2.0 * m1 - m0 - m2)
            delta = (m2 - m0) / denom if abs(denom) > 1e-6 else 0.0
            delta = max(-0.5, min(0.5, delta))

            sub_t = float(i) + delta
            # Normalized position [0.0, 1.0] along line
            norm_t = sub_t / float(num_samples - 1)
            norm_t = max(0.0, min(1.0, norm_t))

            # Real image coordinates of sub-pixel edge
            edge_x = x1 + norm_t * (x2 - x1)
            edge_y = y1 + norm_t * (y2 - y1)

            # Polarity: sign of raw gradient
            polarity = "rising" if grad[i] >= 0 else "falling"

            detected_edges.append({
                'sub_idx': sub_t,
                'norm_t': norm_t,
                'pos': (float(edge_x), float(edge_y)),
                'polarity': polarity,
                'magnitude': float(m1)
            })

    if not detected_edges:
        return {'success': False, 'edges': [], 'distance': 0.0, 'p1': p1, 'p2': p2}

    # If 2 or more edges found, measured distance between first and last edge
    if len(detected_edges) >= 2:
        first_e = detected_edges[0]
        last_e = detected_edges[-1]
        dist_px = math.hypot(last_e['pos'][0] - first_e['pos'][0], last_e['pos'][1] - first_e['pos'][1])
    else:
        dist_px = 0.0

    return {
        'success': True,
        'edges': detected_edges,
        'distance': dist_px,
        'p1': p1,
        'p2': p2,
        'line_length': line_len
    }
