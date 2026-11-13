import math
import cv2
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QPolygon

from .gui_signal import gui_signal
from utils.metrology import fit_circle_3p, compute_angle_deg, run_1d_edge_caliper


class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mouse_pos = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setText("Right-click to interact")
        self.real_resolution = (0, 0) 
        self.zoom_offset = (0, 0)

        # for drag tools (zoom, length, square, circle drag, caliper, calibrate, template)
        self.selection_start = None
        self.selection_end = None
        self.is_dragging = False

        # for multi-click tools
        self.polygon_points = []
        self.circle_points = []
        self.angle_points = []

    def cancel_all_in_progress(self):
        self.polygon_points = []
        self.circle_points = []
        self.angle_points = []
        self.is_dragging = False
        self.selection_start = None
        self.selection_end = None
        self.update()

    def cancel_polygon(self):
        self.cancel_all_in_progress()

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            super().mousePressEvent(event)
            return

        pix_size = pixmap.size()
        lbl_size = self.size()
        x_off = (lbl_size.width() - pix_size.width()) // 2
        y_off = (lbl_size.height() - pix_size.height()) // 2
        image_rect = QRect(x_off, y_off, pix_size.width(), pix_size.height())

        mode = getattr(self.parent(), 'current_tool', 'zoom')

        if event.button() == Qt.LeftButton:
            if image_rect.contains(event.pos()):
                if mode == "pick_color":
                    self.handle_color_pick(event.pos())
                    return
                elif mode == "polygon":
                    self.handle_polygon_click(event.pos())
                    return
                elif mode == "circle_3p":
                    self.handle_circle_3p_click(event.pos())
                    return
                elif mode == "angle":
                    self.handle_angle_click(event.pos())
                    return
                else:
                    self.selection_start = event.pos()
                    self.selection_end = event.pos()
                    self.is_dragging = True

        elif event.button() == Qt.RightButton:
            if mode == "polygon" and self.polygon_points:
                if len(self.polygon_points) >= 3:
                    self.finish_polygon()
                else:
                    self.cancel_all_in_progress()
                event.accept()
                return
            elif mode == "circle_3p" and self.circle_points:
                self.cancel_all_in_progress()
                event.accept()
                return
            elif mode == "angle" and self.angle_points:
                self.cancel_all_in_progress()
                event.accept()
                return

        super().mousePressEvent(event)

    def handle_color_pick(self, pos):
        parent = self.parent()
        if parent and getattr(parent, 'current_frame', None) is not None:
            rx, ry = self.map_to_real(pos)
            h, w = parent.current_frame.shape[:2]
            if 0 <= rx < w and 0 <= ry < h:
                pixel = parent.current_frame[ry, rx]
                b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
                hex_str = f"#{r:02X}{g:02X}{b:02X}"
                print(f"Sampled color at [{rx}, {ry}]: {hex_str} (RGB: {r}, {g}, {b})")
                cb = getattr(parent, '_color_pick_callback', None)
                if cb and callable(cb):
                    cb(hex_str)
                if hasattr(parent, 'stop_color_pick'):
                    parent.stop_color_pick()
                else:
                    parent.current_tool = getattr(parent, '_previous_tool', 'zoom')
                self.update()

    def handle_polygon_click(self, pos):
        if not self.polygon_points:
            self.polygon_points.append(pos)
        else:
            # Check if clicked near start point to close
            start_pt = self.polygon_points[0]
            dist = np.hypot(pos.x() - start_pt.x(), pos.y() - start_pt.y())
            if dist < 12 and len(self.polygon_points) >= 3:
                self.finish_polygon()
                return
            self.polygon_points.append(pos)
        self.update()

    def handle_circle_3p_click(self, pos):
        self.circle_points.append(pos)
        if len(self.circle_points) == 3:
            real_pts = [self.map_to_real(p) for p in self.circle_points]
            res = fit_circle_3p(real_pts[0], real_pts[1], real_pts[2])
            if res is not None:
                cx, cy, r = res
                if r >= 2.0:
                    measurement = {
                        'type': 'circle',
                        'center': (cx, cy),
                        'radius': r,
                        'diameter': r * 2.0,
                        'area': float(math.pi * (r ** 2)),
                        'circumference': float(2.0 * math.pi * r),
                        'points': real_pts,
                        'method': '3point'
                    }
                    parent = self.parent()
                    if hasattr(parent, 'measurements'):
                        parent.measurements.append(measurement)
            self.circle_points = []
        self.update()

    def handle_angle_click(self, pos):
        self.angle_points.append(pos)
        if len(self.angle_points) == 3:
            p1_real = self.map_to_real(self.angle_points[0])
            v_real = self.map_to_real(self.angle_points[1])
            p2_real = self.map_to_real(self.angle_points[2])
            angle_deg = compute_angle_deg(p1_real, v_real, p2_real)
            if angle_deg > 0.01:
                measurement = {
                    'type': 'angle',
                    'p1': p1_real,
                    'vertex': v_real,
                    'p2': p2_real,
                    'angle': angle_deg
                }
                parent = self.parent()
                if hasattr(parent, 'measurements'):
                    parent.measurements.append(measurement)
            self.angle_points = []
        self.update()

    def mouseDoubleClickEvent(self, event):
        mode = getattr(self.parent(), 'current_tool', 'zoom')
        if mode == "polygon":
            if len(self.polygon_points) >= 3:
                self.finish_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_S or event.text().lower() == 's') and (event.modifiers() & Qt.ControlModifier):
            parent = self.parent()
            if parent and hasattr(parent, 'main_window') and hasattr(parent.main_window, 'capture_image'):
                parent.main_window.capture_image()
            event.accept()
            return

        mode = getattr(self.parent(), 'current_tool', 'zoom')
        if mode == "polygon" and self.polygon_points:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if len(self.polygon_points) >= 3:
                    self.finish_polygon()
                return
            elif event.key() == Qt.Key_Escape:
                self.cancel_all_in_progress()
                return
        elif mode == "template":
            if event.key() == Qt.Key_Escape:
                parent = self.parent()
                if hasattr(parent, 'stop_template_highlight'):
                    parent.stop_template_highlight()
                event.accept()
                return
        elif mode == "crop_drag":
            if event.key() == Qt.Key_Escape:
                parent = self.parent()
                if hasattr(parent, 'stop_crop_drag'):
                    parent.stop_crop_drag()
                event.accept()
                return
        elif mode == "pick_color":
            if event.key() == Qt.Key_Escape:
                parent = self.parent()
                if hasattr(parent, 'stop_color_pick'):
                    parent.stop_color_pick()
                event.accept()
                return
        elif mode in ("circle_3p", "angle") and (self.circle_points or self.angle_points):
            if event.key() == Qt.Key_Escape:
                self.cancel_all_in_progress()
                return
        elif event.key() == Qt.Key_Escape:
            self.cancel_all_in_progress()
            parent = self.parent()
            if hasattr(parent, 'is_enlarged') and parent.is_enlarged:
                parent.toggle_enlarge()
                event.accept()
                return
            elif hasattr(parent, 'is_fullscreen_window') and parent.is_fullscreen_window:
                parent.exit_fullscreen_window()
                event.accept()
                return
        super().keyPressEvent(event)

    def finish_polygon(self):
        if len(self.polygon_points) >= 3:
            real_pts = [self.map_to_real(pt) for pt in self.polygon_points]
            pts_arr = np.array(real_pts, dtype=np.float32)
            area = float(cv2.contourArea(pts_arr))
            
            if area >= 2:
                measurement = {
                    'type': 'polygon',
                    'real_points': real_pts,
                    'area': area
                }
                parent = self.parent()
                if hasattr(parent, 'measurements'):
                    parent.measurements.append(measurement)
        self.polygon_points = []
        self.update()

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        pixmap = self.pixmap()
        if self.is_dragging and pixmap and not pixmap.isNull():
            pix_size = pixmap.size()
            lbl_size = self.size()
            x_off = (lbl_size.width() - pix_size.width()) // 2
            y_off = (lbl_size.height() - pix_size.height()) // 2
            
            # Constrain end point to image boundaries
            mx = max(x_off, min(event.pos().x(), x_off + pix_size.width()))
            my = max(y_off, min(event.pos().y(), y_off + pix_size.height()))
            self.selection_end = QPoint(mx, my)
            
        parent = self.parent()
        if parent and hasattr(parent, 'coord_val_label'):
            tx, ty = self.map_to_real(event.pos())
            parent.coord_val_label.setText(f"Pos: [{tx}, {ty}]")

        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            if self.selection_start and self.selection_end and self.parent():
                if hasattr(self.parent(), 'handle_selection'):
                    self.parent().handle_selection(self.selection_start, self.selection_end)
            self.selection_start = None
            self.selection_end = None
            self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.mouse_pos = None
        parent = self.parent()
        if parent and hasattr(parent, 'coord_val_label'):
            parent.coord_val_label.setText("Pos: [0, 0]")
        self.update()
        super().leaveEvent(event)

    def format_length_val(self, px_len):
        parent = self.parent()
        if parent and hasattr(parent, 'format_length'):
            return parent.format_length(px_len)
        return f"{px_len:.2f} px"

    def format_area_val(self, px_area):
        parent = self.parent()
        if parent and hasattr(parent, 'format_area'):
            return parent.format_area(px_area)
        return f"{px_area:.2f} px²"

    def _draw_angle_arc(self, painter, v, p1, p2, radius=24):
        """Draws a smooth circular arc from ray v->p1 to ray v->p2 around vertex v"""
        dx1, dy1 = p1.x() - v.x(), p1.y() - v.y()
        dx2, dy2 = p2.x() - v.x(), p2.y() - v.y()
        ang1 = math.atan2(dy1, dx1)
        ang2 = math.atan2(dy2, dx2)

        diff = ang2 - ang1
        while diff <= -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi

        steps = 14
        arc_pts = []
        for k in range(steps + 1):
            t = float(k) / float(steps)
            a = ang1 + t * diff
            ax = int(round(v.x() + radius * math.cos(a)))
            ay = int(round(v.y() + radius * math.sin(a)))
            arc_pts.append(QPoint(ax, ay))

        for k in range(len(arc_pts) - 1):
            painter.drawLine(arc_pts[k], arc_pts[k + 1])

        mid_a = ang1 + 0.5 * diff
        mid_x = int(round(v.x() + (radius + 8) * math.cos(mid_a)))
        mid_y = int(round(v.y() + (radius + 8) * math.sin(mid_a)))
        return QPoint(mid_x, mid_y)

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pix_size = pixmap.size()
        lbl_size = self.size()
        x_off = (lbl_size.width() - pix_size.width()) // 2
        y_off = (lbl_size.height() - pix_size.height()) // 2
        image_rect = QRect(x_off, y_off, pix_size.width(), pix_size.height())

        # Crosshair cursor guides
        if self.mouse_pos and image_rect.contains(self.mouse_pos):
            guide_pen = QPen(QColor(255, 255, 255, 100), 1)
            guide_pen.setStyle(Qt.DashLine)
            painter.setPen(guide_pen)
            
            mx, my = self.mouse_pos.x(), self.mouse_pos.y()
            painter.drawLine(mx, image_rect.top(), mx, image_rect.bottom())
            painter.drawLine(image_rect.left(), my, image_rect.right(), my)
            tx, ty = self.map_to_real(self.mouse_pos)
            self.last_true_pos = (tx, ty)
        else:
            self.last_true_pos = (0, 0)

        painter.setClipRect(image_rect)

        # Draw completed measurements
        measurements = getattr(self.parent(), 'measurements', [])
        for m in measurements:
            m_type = m.get('type')

            if m_type == "length":
                p1 = self.real_to_screen(m['real_p1'])
                p2 = self.real_to_screen(m['real_p2'])
                color = QColor(255, 0, 255, 200) # Magenta
                painter.setPen(QPen(color, 2))
                painter.drawLine(p1, p2)
                # End ticks
                painter.setBrush(color)
                painter.drawEllipse(p1, 3, 3)
                painter.drawEllipse(p2, 3, 3)
                if image_rect.contains(p2):
                    label_str = self.format_length_val(m['length'])
                    self._draw_label(painter, p2, label_str, color)

            elif m_type in ("square", "area"):
                p1 = self.real_to_screen(m['real_p1'])
                p2 = self.real_to_screen(m['real_p2'])
                color = QColor(255, 255, 0, 200)
                painter.setPen(QPen(color, 1))
                painter.setBrush(QColor(255, 255, 0, 30))
                painter.drawRect(QRect(p1, p2).normalized())
                if image_rect.contains(p1):
                    label_str = self.format_area_val(m['area'])
                    self._draw_label(painter, p1, label_str, color)

            elif m_type == "polygon":
                real_pts = m.get('real_points', [])
                if len(real_pts) >= 3:
                    screen_pts = [self.real_to_screen(rp) for rp in real_pts]
                    color = QColor(0, 230, 255, 200) # Bright cyan
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(QColor(0, 230, 255, 40))
                    
                    poly = QPolygon(screen_pts)
                    painter.drawPolygon(poly)
                    
                    painter.setBrush(color)
                    for pt in screen_pts:
                        painter.drawEllipse(pt, 3, 3)
                    
                    cx = sum(p.x() for p in screen_pts) // len(screen_pts)
                    cy = sum(p.y() for p in screen_pts) // len(screen_pts)
                    label_pos = QPoint(cx, cy)
                    if image_rect.contains(label_pos):
                        label_str = self.format_area_val(m['area'])
                        self._draw_label(painter, label_pos, label_str, color)

            elif m_type == "circle":
                c_screen = self.real_to_screen(m['center'])
                r_real = m['radius']
                edge_screen = self.real_to_screen((m['center'][0] + r_real, m['center'][1]))
                r_screen = max(1, abs(edge_screen.x() - c_screen.x()))

                color = QColor(0, 230, 118, 220) # Emerald green
                painter.setPen(QPen(color, 2))
                painter.setBrush(QColor(0, 230, 118, 25))
                painter.drawEllipse(c_screen, r_screen, r_screen)

                # Center crosshair
                painter.setPen(QPen(color, 1))
                painter.drawLine(c_screen.x() - 6, c_screen.y(), c_screen.x() + 6, c_screen.y())
                painter.drawLine(c_screen.x(), c_screen.y() - 6, c_screen.x(), c_screen.y() + 6)

                # Radius line
                painter.setPen(QPen(color, 1, Qt.DashLine))
                painter.drawLine(c_screen, QPoint(c_screen.x() + r_screen, c_screen.y()))

                # If 3-point fit, show sampled perimeter dots
                if m.get('points'):
                    painter.setBrush(QColor(255, 255, 255))
                    painter.setPen(QPen(color, 1))
                    for pt_real in m['points']:
                        sp = self.real_to_screen(pt_real)
                        painter.drawEllipse(sp, 3, 3)

                r_str = self.format_length_val(r_real)
                d_str = self.format_length_val(r_real * 2.0)
                a_str = self.format_area_val(math.pi * (r_real ** 2))
                badge_pos = QPoint(c_screen.x() + r_screen // 2, c_screen.y() - 8)
                if image_rect.contains(c_screen):
                    self._draw_label(painter, badge_pos, f"R: {r_str} | D: {d_str} | A: {a_str}", color)

            elif m_type == "angle":
                v_screen = self.real_to_screen(m['vertex'])
                p1_screen = self.real_to_screen(m['p1'])
                p2_screen = self.real_to_screen(m['p2'])
                color = QColor(255, 171, 0, 220) # Amber orange

                painter.setPen(QPen(color, 2))
                painter.drawLine(v_screen, p1_screen)
                painter.drawLine(v_screen, p2_screen)

                # Vertex dot
                painter.setBrush(color)
                painter.drawEllipse(v_screen, 4, 4)
                painter.drawEllipse(p1_screen, 3, 3)
                painter.drawEllipse(p2_screen, 3, 3)

                # Angle arc and badge
                arc_r = min(26, max(12, int(min(np.hypot(p1_screen.x()-v_screen.x(), p1_screen.y()-v_screen.y()),
                                                np.hypot(p2_screen.x()-v_screen.x(), p2_screen.y()-v_screen.y())) * 0.4)))
                mid_arc = self._draw_angle_arc(painter, v_screen, p1_screen, p2_screen, radius=arc_r)
                if image_rect.contains(v_screen):
                    self._draw_label(painter, mid_arc, f"θ: {m['angle']:.1f}°", color)

            elif m_type == "caliper":
                p1 = self.real_to_screen(m['real_p1'])
                p2 = self.real_to_screen(m['real_p2'])
                color = QColor(0, 210, 255, 220) # Neon blue

                # Baseline
                painter.setPen(QPen(color, 1, Qt.DashLine))
                painter.drawLine(p1, p2)

                dx = float(p2.x() - p1.x())
                dy = float(p2.y() - p1.y())
                L = math.hypot(dx, dy)
                if L > 1e-3:
                    nx = -dy / L
                    ny = dx / L

                    # Draw detected sub-pixel edges as caliper jaws
                    edges = m.get('edges', [])
                    for ep_real in edges:
                        ep_s = self.real_to_screen(ep_real)
                        j1 = QPoint(int(round(ep_s.x() + 10 * nx)), int(round(ep_s.y() + 10 * ny)))
                        j2 = QPoint(int(round(ep_s.x() - 10 * nx)), int(round(ep_s.y() - 10 * ny)))
                        painter.setPen(QPen(color, 2))
                        painter.drawLine(j1, j2)
                        painter.setBrush(QColor(255, 255, 255))
                        painter.drawEllipse(ep_s, 2, 2)

                    # Distance badge if >= 2 edges
                    if len(edges) >= 2 and m.get('distance', 0) > 0:
                        ep1_s = self.real_to_screen(edges[0])
                        ep2_s = self.real_to_screen(edges[-1])
                        mid_s = QPoint((ep1_s.x() + ep2_s.x()) // 2, (ep1_s.y() + ep2_s.y()) // 2)
                        dist_str = self.format_length_val(m['distance'])
                        self._draw_label(painter, mid_s, f"Caliper: {dist_str}", color)
                    elif len(edges) == 1:
                        ep1_s = self.real_to_screen(edges[0])
                        self._draw_label(painter, ep1_s, f"Edge @ ({edges[0][0]:.1f}, {edges[0][1]:.1f})", color)

        # In-progress interactive tool rendering
        mode = getattr(self.parent(), 'current_tool', 'zoom')

        # 1. In-progress Polygon
        if mode == "polygon" and self.polygon_points:
            color = QColor(0, 230, 255)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            for i in range(len(self.polygon_points)):
                pt = self.polygon_points[i]
                painter.drawEllipse(pt, 3, 3)
                if i > 0:
                    painter.drawLine(self.polygon_points[i - 1], pt)
            
            if self.mouse_pos and image_rect.contains(self.mouse_pos):
                dash_pen = QPen(color, 2, Qt.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(self.polygon_points[-1], self.mouse_pos)
                if len(self.polygon_points) >= 2:
                    start_pt = self.polygon_points[0]
                    dist_to_start = np.hypot(self.mouse_pos.x() - start_pt.x(), self.mouse_pos.y() - start_pt.y())
                    if dist_to_start < 12 and len(self.polygon_points) >= 3:
                        painter.setPen(QPen(QColor(255, 50, 50), 2))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(start_pt, 7, 7)
                    else:
                        faint_pen = QPen(QColor(0, 230, 255, 100), 1, Qt.DotLine)
                        painter.setPen(faint_pen)
                        painter.drawLine(self.mouse_pos, start_pt)

        # 2. In-progress 3-Point Circle
        elif mode == "circle_3p" and self.circle_points:
            color = QColor(0, 230, 118)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            for pt in self.circle_points:
                painter.drawEllipse(pt, 4, 4)

            if len(self.circle_points) == 1 and self.mouse_pos and image_rect.contains(self.mouse_pos):
                painter.setPen(QPen(color, 1, Qt.DashLine))
                painter.drawLine(self.circle_points[0], self.mouse_pos)
                self._draw_label(painter, self.mouse_pos, "Point 2 (Circumference)", color)

            elif len(self.circle_points) == 2 and self.mouse_pos and image_rect.contains(self.mouse_pos):
                p1_s = self.circle_points[0]
                p2_s = self.circle_points[1]
                p3_s = self.mouse_pos
                painter.setPen(QPen(color, 1, Qt.DotLine))
                painter.drawLine(p1_s, p2_s)
                painter.drawLine(p2_s, p3_s)

                # Live circumcircle preview
                res = fit_circle_3p((p1_s.x(), p1_s.y()), (p2_s.x(), p2_s.y()), (p3_s.x(), p3_s.y()))
                if res is not None:
                    cx, cy, r = res
                    if 1 < r < 2000:
                        painter.setPen(QPen(color, 2, Qt.DashLine))
                        painter.setBrush(QColor(0, 230, 118, 20))
                        painter.drawEllipse(QPoint(int(round(cx)), int(round(cy))), int(round(r)), int(round(r)))
                self._draw_label(painter, p3_s, "Point 3 (Click to fit circle)", color)

        # 3. In-progress Angle / Protractor
        elif mode == "angle" and self.angle_points:
            color = QColor(255, 171, 0)
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            for pt in self.angle_points:
                painter.drawEllipse(pt, 4, 4)

            if len(self.angle_points) == 1 and self.mouse_pos and image_rect.contains(self.mouse_pos):
                painter.setPen(QPen(color, 1, Qt.DashLine))
                painter.drawLine(self.angle_points[0], self.mouse_pos)
                self._draw_label(painter, self.mouse_pos, "Click Vertex (Apex)", color)

            elif len(self.angle_points) == 2 and self.mouse_pos and image_rect.contains(self.mouse_pos):
                p1_s = self.angle_points[0]
                v_s = self.angle_points[1]
                p2_s = self.mouse_pos

                painter.setPen(QPen(color, 2))
                painter.drawLine(v_s, p1_s)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawLine(v_s, p2_s)

                # Live angle preview
                deg = compute_angle_deg((p1_s.x(), p1_s.y()), (v_s.x(), v_s.y()), (p2_s.x(), p2_s.y()))
                arc_r = min(24, max(12, int(min(np.hypot(p1_s.x()-v_s.x(), p1_s.y()-v_s.y()),
                                                np.hypot(p2_s.x()-v_s.x(), p2_s.y()-v_s.y())) * 0.4)))
                mid_arc = self._draw_angle_arc(painter, v_s, p1_s, p2_s, radius=arc_r)
                self._draw_label(painter, mid_arc, f"θ: {deg:.1f}° (Click Point 2)", color)

        # 4. In-progress Drag Tools (Circle drag, Caliper drag, Calibrate drag, Length, Area, Template, Zoom)
        elif self.is_dragging and self.selection_start and self.selection_end:
            if mode == "circle":
                color = QColor(0, 230, 118)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.setBrush(QColor(0, 230, 118, 25))
                c = self.selection_start
                e = self.selection_end
                r_screen = int(round(math.hypot(e.x() - c.x(), e.y() - c.y())))
                painter.drawEllipse(c, r_screen, r_screen)

                # Center crosshair and radius line
                painter.setPen(QPen(color, 1))
                painter.drawLine(c.x() - 6, c.y(), c.x() + 6, c.y())
                painter.drawLine(c.x(), c.y() - 6, c.x(), c.y() + 6)
                painter.drawLine(c, e)

                c_real = self.map_to_real(c)
                e_real = self.map_to_real(e)
                r_real = math.hypot(e_real[0] - c_real[0], e_real[1] - c_real[1])
                r_str = self.format_length_val(r_real)
                d_str = self.format_length_val(r_real * 2.0)
                a_str = self.format_area_val(math.pi * (r_real ** 2))
                self._draw_label(painter, e, f"R: {r_str} | D: {d_str} | A: {a_str}", color)

            elif mode == "caliper":
                color = QColor(0, 210, 255)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                p1, p2 = self.selection_start, self.selection_end
                painter.drawLine(p1, p2)

                dx = float(p2.x() - p1.x())
                dy = float(p2.y() - p1.y())
                L = math.hypot(dx, dy)
                if L > 1e-3:
                    nx = -dy / L
                    ny = dx / L
                    painter.setPen(QPen(color, 2))
                    painter.drawLine(QPoint(int(round(p1.x() + 8*nx)), int(round(p1.y() + 8*ny))),
                                     QPoint(int(round(p1.x() - 8*nx)), int(round(p1.y() - 8*ny))))
                    painter.drawLine(QPoint(int(round(p2.x() + 8*nx)), int(round(p2.y() + 8*ny))),
                                     QPoint(int(round(p2.x() - 8*nx)), int(round(p2.y() - 8*ny))))

                rp1 = self.map_to_real(p1)
                rp2 = self.map_to_real(p2)
                scan_len = math.hypot(rp2[0] - rp1[0], rp2[1] - rp1[1])
                self._draw_label(painter, p2, f"Caliper Scan: {self.format_length_val(scan_len)}", color)

            elif mode == "calibrate":
                color = QColor(255, 215, 0) # Gold
                painter.setPen(QPen(color, 2))
                p1, p2 = self.selection_start, self.selection_end
                painter.drawLine(p1, p2)

                dx = float(p2.x() - p1.x())
                dy = float(p2.y() - p1.y())
                L = math.hypot(dx, dy)
                if L > 1e-3:
                    nx = -dy / L
                    ny = dx / L
                    painter.drawLine(QPoint(int(round(p1.x() + 8*nx)), int(round(p1.y() + 8*ny))),
                                     QPoint(int(round(p1.x() - 8*nx)), int(round(p1.y() - 8*ny))))
                    painter.drawLine(QPoint(int(round(p2.x() + 8*nx)), int(round(p2.y() + 8*ny))),
                                     QPoint(int(round(p2.x() - 8*nx)), int(round(p2.y() - 8*ny))))

                rp1 = self.map_to_real(p1)
                rp2 = self.map_to_real(p2)
                dist = math.hypot(rp2[0] - rp1[0], rp2[1] - rp1[1])
                self._draw_label(painter, p2, f"Calibration Ref: {dist:.1f} px", color)

            elif mode == "template":
                tpl_color = QColor(0, 220, 255) # Cyan
                painter.setPen(QPen(tpl_color, 2, Qt.DashLine))
                painter.setBrush(QColor(0, 220, 255, 45))
                drag_rect = QRect(self.selection_start, self.selection_end).normalized()
                painter.drawRect(drag_rect)

                p1_real = self.map_to_real(self.selection_start)
                p2_real = self.map_to_real(self.selection_end)
                rw = abs(p2_real[0] - p1_real[0])
                rh = abs(p2_real[1] - p1_real[1])
                self._draw_label(painter, drag_rect.bottomRight(), f"Template: {rw} x {rh} px", tpl_color)

            elif mode == "crop_drag":
                crop_color = QColor(50, 205, 50) # Lime Green
                painter.setPen(QPen(crop_color, 2, Qt.DashLine))
                painter.setBrush(QColor(50, 205, 50, 45))
                drag_rect = QRect(self.selection_start, self.selection_end).normalized()
                painter.drawRect(drag_rect)

                p1_real = self.map_to_real(self.selection_start)
                p2_real = self.map_to_real(self.selection_end)
                rw = abs(p2_real[0] - p1_real[0])
                rh = abs(p2_real[1] - p1_real[1])
                self._draw_label(painter, drag_rect.bottomRight(), f"Crop: {rw} x {rh} px", crop_color)

            elif mode == "length":
                color = QColor(255, 0, 255)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawLine(self.selection_start, self.selection_end)
                rp1 = self.map_to_real(self.selection_start)
                rp2 = self.map_to_real(self.selection_end)
                dist = math.hypot(rp2[0] - rp1[0], rp2[1] - rp1[1])
                self._draw_label(painter, self.selection_end, self.format_length_val(dist), color)

            elif mode in ("area", "square", "zoom"):
                color = QColor(255, 255, 0)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.setBrush(QColor(color.red(), color.green(), 0, 40))
                drag_rect = QRect(self.selection_start, self.selection_end).normalized()
                painter.drawRect(drag_rect)
                if mode != "zoom":
                    rp1 = self.map_to_real(self.selection_start)
                    rp2 = self.map_to_real(self.selection_end)
                    area = abs(rp2[0] - rp1[0]) * abs(rp2[1] - rp1[1])
                    self._draw_label(painter, drag_rect.bottomRight(), self.format_area_val(area), color)

        painter.end()

    def _draw_label(self, painter, pos, text, color):
        """Helper to draw text with a small dark background for visibility"""
        painter.save()
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        metrics = painter.fontMetrics()
        rect = metrics.boundingRect(text)
        rect.moveTopLeft(QPoint(pos.x() + 5, pos.y() - 15))
        rect.adjust(-2, -2, 2, 2)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(rect, 2, 2)
        
        painter.setPen(Qt.white)
        painter.drawText(pos.x() + 5, pos.y() - 5, text)
        painter.restore()

    def map_to_real(self, pos):
        """Maps mouse click position on UI label directly to real image pixel coordinates."""
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            return 0, 0

        pix_size = pixmap.size()
        lbl_size = self.size()
        
        # Calculate offset caused by Qt.KeepAspectRatio centering
        x_off = (lbl_size.width() - pix_size.width()) // 2
        y_off = (lbl_size.height() - pix_size.height()) // 2
        
        rel_x = pos.x() - x_off
        rel_y = pos.y() - y_off
        
        # Determine visible source field (full or zoom crop)
        parent = self.parent()
        zoom_rect = getattr(parent, 'zoom_rect', None)
        if zoom_rect:
            _, _, active_w, active_h = zoom_rect
        else:
            active_w, active_h = self.real_resolution
            if active_w <= 0 or active_h <= 0:
                active_w, active_h = pix_size.width(), pix_size.height()

        scale_x = active_w / float(pix_size.width()) if pix_size.width() > 0 else 1.0
        scale_y = active_h / float(pix_size.height()) if pix_size.height() > 0 else 1.0
        
        real_x = int(rel_x * scale_x) + self.zoom_offset[0]
        real_y = int(rel_y * scale_y) + self.zoom_offset[1]

        if self.real_resolution[0] > 0 and self.real_resolution[1] > 0:
            real_x = max(0, min(real_x, self.real_resolution[0] - 1))
            real_y = max(0, min(real_y, self.real_resolution[1] - 1))
        
        return real_x, real_y
    
    def real_to_screen(self, real_pt):
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull():
            return QPoint(0, 0)
        
        rx, ry = real_pt
        ox, oy = self.zoom_offset
        
        parent = self.parent()
        zoom_rect = getattr(parent, 'zoom_rect', None)
        if zoom_rect:
            _, _, lw, lh = zoom_rect
        else:
            lw, lh = self.real_resolution
            if lw <= 0 or lh <= 0:
                lw, lh = pixmap.size().width(), pixmap.size().height()
            
        pix_size = pixmap.size()
        lbl_size = self.size()
        
        scale_x = pix_size.width() / lw if lw > 0 else 1
        scale_y = pix_size.height() / lh if lh > 0 else 1
        
        x_off = (lbl_size.width() - pix_size.width()) // 2
        y_off = (lbl_size.height() - pix_size.height()) // 2
        
        sx = int((rx - ox) * scale_x) + x_off
        sy = int((ry - oy) * scale_y) + y_off

        self.last_image_rect = QRect(x_off, y_off, pix_size.width(), pix_size.height())
        return QPoint(sx, sy)
