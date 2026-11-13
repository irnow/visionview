import os
import sys
import cv2
import time
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QFrame, QLabel, QFileDialog,
                             QMenu, QMenuBar, QStackedWidget, QSplitter)
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Slot, QRect, QRectF, QPoint, QPointF, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIcon

from .gui_signal import gui_signal
from .process_canva import ProcessWindow
from .video_label import VideoLabel
from camera import *

try:
    from qt_material_icons import MaterialIcon
except ImportError:
    MaterialIcon = lambda name: QIcon()

_ICON_CACHE = {}

def get_module_icon(name: str, color: str = "white") -> QIcon:
    key = (name, color)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    try:
        ico = MaterialIcon(name)
        if color:
            res = QIcon(ico.pixmap(32, 32, color=QColor(color)))
        else:
            res = ico
        _ICON_CACHE[key] = res
        return res
    except Exception:
        return QIcon()

class WindowModule(QFrame):
    def __init__(self, main_window, index):
        super().__init__()
        self.main_window = main_window
        self.index = index
        self.is_fullscreen_window = False
        self.is_enlarged = False # Track state for toggle

        if hasattr(self.main_window, 'tool_group'):
            active_action = self.main_window.tool_group.checkedAction()
            self.current_tool = active_action.text().lower() if active_action else "zoom"
        else:
            self.current_tool = "zoom"

        self.thread = None
        self.worker = None
        self.current_frame = None
        self.zoom_rect = None
        self.measurements = []
        
        self.calibration = {
            'scale': 1.0,
            'unit': 'px',
            'is_calibrated': False,
            'ref_pixels': 0.0,
            'ref_real': 0.0
        }
        if hasattr(self.main_window, 'global_calibration') and self.main_window.global_calibration.get('is_calibrated'):
            self.calibration.update(self.main_window.global_calibration)

        # for record video
        self.is_recording = False
        self.video_writer = None
        self.video_path = None

        self.static_image = False

        self.process_window = None  # Store the window instance here
        self.saved_nodes = []       # Store node data (title, x, y)

        self.gui()

    def gui(self):
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.setObjectName("moduleFrame")
        
        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)

        # --- CUSTOM HEADER BAR ---
        self.header = QFrame()
        self.header.setFixedHeight(30)
        self.header.setStyleSheet("background-color: #444; border-bottom: 1px solid #555;")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(5, 0, 5, 0)

        # 1. Serial/Title Label
        self.title_label = QLabel(f"Module {self.index}")
        self.title_label.setStyleSheet("color: white; font-weight: bold; border: none;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()

        # 2. Enlarge/Minimize Button (Panel Canvas)
        self.btn_enlarge = QPushButton()
        self.btn_enlarge.setIcon(get_module_icon("open_in_full", "white"))
        self.btn_enlarge.setIconSize(QSize(15, 15))
        self.btn_enlarge.setToolTip("Enlarge in Panel Canvas (Esc to return)")
        self.btn_enlarge.setFixedSize(26, 22)
        self.btn_enlarge.setStyleSheet("background: #555; border: 1px solid #777; border-radius: 2px;")
        self.btn_enlarge.clicked.connect(self.toggle_enlarge)
        header_layout.addWidget(self.btn_enlarge)

        # 3. Close Button
        self.btn_close = QPushButton()
        self.btn_close.setIcon(get_module_icon("close", "white"))
        self.btn_close.setIconSize(QSize(15, 15))
        self.btn_close.setToolTip("Close Module")
        self.btn_close.setFixedSize(26, 22)
        self.btn_close.setStyleSheet("background: #c0392b; border: 1px solid #777; border-radius: 2px;")
        self.btn_close.clicked.connect(self.request_close)
        header_layout.addWidget(self.btn_close)

        self.main_layout.addWidget(self.header)

        # --- VIDEO/IMAGE AREA ---
        # self.label = QLabel("Right-click to Interact")
        self.label = VideoLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: none;")
        self.main_layout.addWidget(self.label)

        # --- CUSTOM FOOTER BAR ---
        self.footer = QFrame()
        self.footer.setFixedHeight(25)
        # Give it a slight border-top to separate it from the video
        self.footer.setStyleSheet("QFrame { background-color: #1e1e1e; border-top: 1px solid #333; border-bottom: none; }")
        
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(10, 0, 10, 0)
        footer_layout.setSpacing(10)

        label_style = "color: #aaa; font-size: 11px; border: none; background: none;"

        # 1. Resolution Label
        self.res_val_label = QLabel("0 x 0")
        self.res_val_label.setStyleSheet(label_style)
        
        # 2. Coordinates Label
        self.coord_val_label = QLabel("Pos: [0, 0]")
        self.coord_val_label.setStyleSheet(label_style)

        # 3. FPS Label
        self.fps_val_label = QLabel("FPS: 0.0")
        self.fps_val_label.setStyleSheet("color: #409eff; font-size: 11px; font-weight: bold; border: none; background: none;")

        # Helper create vertical separators
        def get_sep():
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setFixedWidth(1)
            sep.setStyleSheet("background-color: #444; margin: 5px 0px;")
            return sep

        footer_layout.addWidget(self.res_val_label)
        footer_layout.addWidget(get_sep())
        footer_layout.addWidget(self.coord_val_label)

        # Scale calibration status badge in footer
        self.calib_sep = get_sep()
        self.calib_val_label = QLabel("Scale: 1 px = 1.0 px")
        self.calib_val_label.setStyleSheet("color: #64b5f6; font-size: 11px; font-weight: bold; border: none; background: none;")
        self.calib_val_label.setToolTip("Scale calibration (Right-click to calibrate or reset)")
        
        if self.calibration.get('is_calibrated'):
            self.calib_val_label.setText(f"Scale: 1 px = {self.calibration['scale']:.4g} {self.calibration['unit']}")
            self.calib_val_label.show()
            self.calib_sep.show()
        else:
            self.calib_val_label.hide()
            self.calib_sep.hide()

        footer_layout.addWidget(self.calib_sep)
        footer_layout.addWidget(self.calib_val_label)

        footer_layout.addStretch() # This pushes everything after it to the right
        footer_layout.addWidget(get_sep())
        footer_layout.addWidget(self.fps_val_label)

        self.main_layout.addWidget(self.footer)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        init_cam = context_menu.addAction("Start Camera")
        upload_img = context_menu.addAction("Upload Image")
        process = context_menu.addAction("Process")
        calib_scale = context_menu.addAction("Calibrate Scale (px -> mm)...")

        reset_zoom = None
        reset_measure = None
        reset_calib = None

        context_menu.addSeparator()
        
        if self.zoom_rect:
            reset_zoom = context_menu.addAction("Reset Zoom")
        
        if self.measurements:
            reset_measure = context_menu.addAction("Clear Measurements")

        if self.calibration.get('is_calibrated', False):
            reset_calib = context_menu.addAction("Reset Calibration (Use px)")

        context_menu.setStyleSheet(
            "background: white; \
            color: black; \
            selection-background-color: #409eff; \
            selection-color: white;" \
            "")
        
        action = context_menu.exec(event.globalPos())
        
        if action == init_cam:
            self.is_camera()

        elif action == upload_img:
            self.upload_image()

        elif action == process:
            self.open_process_editor()

        elif action == calib_scale:
            self.open_calibration_dialog()

        elif action == reset_calib:
            self.reset_calibration()

        elif action == reset_zoom:
            self.zoom_rect = None
            self.label.zoom_offset = (0, 0)
            self.label.update()

        elif action == reset_measure:
            self.measurements = []
            if hasattr(self.label, 'cancel_all_in_progress'):
                self.label.cancel_all_in_progress()
            elif hasattr(self.label, 'cancel_polygon'):
                self.label.cancel_polygon()
            self.label.update()

    def is_camera(self):
        dialog = CameraInitDialog(self)
        if dialog.exec():
            camera_obj = dialog.selected_camera_instance    # Class
            device_info = dialog.selected_device            # Device instance
            serial_info = dialog.selected_sn                # Device name (or serial number)
            self.serial_info = serial_info

            if serial_info in self.main_window.active_cameras:
                print("Camera in use!")
                return
            else:
                gui_signal.update_active_cameras.emit(serial_info)

            self.title_label.setText(f"{serial_info}")
            
            camera_obj.device = device_info
            self.label.setText(f"Starting Serial: {serial_info}...")
            
            self.start_camera(camera_obj, device_info)
            
    def enter_fullscreen_window(self):
        """Detaches selected module as a full screen window on the desktop (only image shown, no header/footer)"""
        if getattr(self, 'is_fullscreen_window', False):
            return
        self.is_fullscreen_window = True

        # Hide header and footer so ONLY the image is shown
        if hasattr(self, 'header') and self.header:
            self.header.hide()
        if hasattr(self, 'footer') and self.footer:
            self.footer.hide()

        # Clean black styling without borders
        self._saved_style = self.styleSheet()
        self.setStyleSheet("background-color: black; border: none;")

        self.setParent(None)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

        # Ensure focus is on the video label so Esc is immediately received
        if hasattr(self, 'label'):
            self.label.setFocus()

        if hasattr(self.main_window, 'action_fullscreen_module'):
            self.main_window.action_fullscreen_module.setIcon(MaterialIcon("fullscreen_exit"))
            self.main_window.action_fullscreen_module.setText("Exit Full Screen")
            self.main_window.action_fullscreen_module.setToolTip("Exit Full Screen (Esc to return)")

    def exit_fullscreen_window(self):
        """Restores module back into the central grid layout with header and footer restored"""
        if not getattr(self, 'is_fullscreen_window', False):
            return
        self.is_fullscreen_window = False

        # Restore header and footer
        if hasattr(self, 'header') and self.header:
            self.header.show()
        if hasattr(self, 'footer') and self.footer:
            self.footer.show()

        # Restore normal stylesheet
        saved_style = getattr(self, '_saved_style', None)
        if saved_style:
            self.setStyleSheet(saved_style)
        else:
            self.setStyleSheet("background-color: #2b2b2b; border: 2px solid #00aaff;")

        self.setParent(self.main_window.central_container)
        self.setWindowFlags(Qt.Widget)
        if hasattr(self.main_window, 'action_fullscreen_module'):
            self.main_window.action_fullscreen_module.setIcon(MaterialIcon("open_in_full"))
            self.main_window.action_fullscreen_module.setText("Full Screen Selected Module")
            self.main_window.action_fullscreen_module.setToolTip("Full Screen Selected Module on Desktop/Window (Esc to return)")
        self.main_window.update_grid_layout()
        self.showNormal()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def toggle_fullscreen_window(self):
        if getattr(self, 'is_fullscreen_window', False):
            self.exit_fullscreen_window()
        else:
            self.enter_fullscreen_window()

    def toggle_enlarge(self):
        self.is_enlarged = not self.is_enlarged
        if self.is_enlarged:
            self.btn_enlarge.setIcon(get_module_icon("fullscreen_exit", "white"))
            self.btn_enlarge.setToolTip("Restore Grid (Esc)")
        else:
            self.btn_enlarge.setIcon(get_module_icon("open_in_full", "white"))
            self.btn_enlarge.setToolTip("Enlarge in Panel Canvas (Esc to return)")
        gui_signal.enlarge_module.emit(self, self.is_enlarged)

    def mousePressEvent(self, event):
        gui_signal.module_selected.emit(self)
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_S or event.text().lower() == 's') and (event.modifiers() & Qt.ControlModifier):
            if hasattr(self.main_window, 'capture_image'):
                self.main_window.capture_image()
            elif getattr(self, 'current_frame', None) is not None:
                save_dir = getattr(self.main_window, 'dir_path', '')
                if save_dir:
                    self.capture_image(save_dir)
            event.accept()
            return

        if event.key() == Qt.Key_Escape:
            if getattr(self, 'is_enlarged', False):
                self.toggle_enlarge()
                event.accept()
                return
            if getattr(self, 'is_fullscreen_window', False):
                self.exit_fullscreen_window()
                event.accept()
                return
        super().keyPressEvent(event)

    def start_camera(self, camera_obj, camera_instance):
        # Stop any existing thread first to prevent the "Destroyed" error
        self.close_event_logic()

        self.thread = QThread()
        self.worker = CameraWorker(camera_obj, camera_instance)
        self.worker.moveToThread(self.thread)

        # Connect Signals
        self.thread.started.connect(self.worker.process)
        self.worker.image_ready.connect(self.update_view)
        
        # Ensure thread cleanup when worker finishes
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(lambda e: print(f"Error: {e}"))
        self.worker.fps_ready.connect(self.update_fps_label)
        
        self.thread.start()
        if hasattr(self.main_window, 'set_active_module'):
            self.main_window.set_active_module(self)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if not file_path:
            return

        self.close_event_logic()
        self.remove_from_active_cameras()

        img = cv2.imread(file_path)
        if img is None:
            self.label.setText("Error: Invalid Image File")
            return
        
        from pathlib import Path
        self.title_label.setText(f"{Path(file_path).name}")
        self.static_image = True

        self.thread = QThread()
        self.worker = ImageWorker(img)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.process)
        self.worker.image_ready.connect(self.update_view)
        
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        print(f"Module {self.index}: Started background processing for {file_path}")
        if hasattr(self.main_window, 'set_active_module'):
            self.main_window.set_active_module(self)

    @Slot(np.ndarray, int, int)
    def update_view(self, frame, orig_w, orig_h):
        if frame is None: 
            return
        
        # 1. Store and display true resolution in the footer (only update if changed)
        channels = 1 if len(frame.shape) == 2 else frame.shape[2]
        res_str = f"{orig_w} x {orig_h}" if channels == 1 else f"{orig_w} x {orig_h} x {channels}"
        if getattr(self, '_last_res_str', None) != res_str:
            self._last_res_str = res_str
            self.res_val_label.setText(res_str)
        
        # Store dimensions on the VideoLabel for coordinate transformations
        self.label.real_resolution = (orig_w, orig_h)

        # 2. Prepare frame for display
        display_frame = frame
        height, width = display_frame.shape[:2]

        # Handle Crop / Zoom UI relative to downscaled frame
        if self.zoom_rect:
            zx, zy, zw, zh = self.zoom_rect
            scale_w = width / orig_w if orig_w > 0 else 1.0
            scale_h = height / orig_h if orig_h > 0 else 1.0
            
            crop_x = max(0, min(int(zx * scale_w), width - 1))
            crop_y = max(0, min(int(zy * scale_h), height - 1))
            crop_w = max(1, min(int(zw * scale_w), width - crop_x))
            crop_h = max(1, min(int(zh * scale_h), height - crop_y))
            
            self.label.zoom_offset = (zx, zy)
            display_frame = display_frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        else:
            self.label.zoom_offset = (0, 0)

        # 3. Handle video recording
        if self.is_recording and self.video_writer is not None:
            try:
                rec_frame = display_frame
                if len(rec_frame.shape) == 2:
                    rec_frame = cv2.cvtColor(rec_frame, cv2.COLOR_GRAY2BGR)
                elif rec_frame.shape[2] == 4:
                    rec_frame = cv2.cvtColor(rec_frame, cv2.COLOR_BGRA2BGR)
                
                rh, rw = rec_frame.shape[:2]
                if hasattr(self, 'record_width') and hasattr(self, 'record_height'):
                    if (rw, rh) != (self.record_width, self.record_height):
                        rec_frame = cv2.resize(rec_frame, (self.record_width, self.record_height))
                self.video_writer.write(rec_frame)
            except Exception as e:
                print(f"Video write error: {e}")

        # 4. Save current full-resolution frame for image capture / zoom mapping without extra copy
        self.current_frame = display_frame

        # 5. Fast downsampling for UI rendering using OpenCV SIMD (eliminates GUI thread lag)
        try:
            dh, dw = display_frame.shape[:2]
            lbl_w = self.label.width()
            lbl_h = self.label.height()

            if lbl_w > 10 and lbl_h > 10:
                scale = min(lbl_w / dw, lbl_h / dh)
                if abs(scale - 1.0) > 0.01:
                    tw = max(1, int(round(dw * scale)))
                    th = max(1, int(round(dh * scale)))
                    vis_frame = cv2.resize(display_frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                else:
                    vis_frame = display_frame
            else:
                vis_frame = display_frame

            if not vis_frame.flags['C_CONTIGUOUS']:
                vis_frame = np.ascontiguousarray(vis_frame)

            vh, vw = vis_frame.shape[:2]
            v_bpl = vis_frame.strides[0]
            q_format = QImage.Format_Grayscale8 if len(vis_frame.shape) == 2 else QImage.Format_BGR888

            img = QImage(vis_frame.data, vw, vh, v_bpl, q_format)
            pixmap = QPixmap.fromImage(img)
            self.label.setPixmap(pixmap)
        except Exception as e:
            print(f"Display Error: {e}")

    def update_fps_label(self, fps):
        self.fps_val_label.setText(f"FPS: {fps}")
        
        if hasattr(self.label, 'last_true_pos'):
            tx, ty = self.label.last_true_pos
            self.coord_val_label.setText(f"Pos: [{tx}, {ty}]")

    def handle_selection(self, start_pos, end_pos):
        if self.current_tool == "zoom":
            self.apply_zoom(start_pos, end_pos)
        elif self.current_tool == "template":
            self.capture_template_from_roi(start_pos, end_pos)
        elif self.current_tool == "crop_drag":
            self.capture_crop_from_roi(start_pos, end_pos)
        elif self.current_tool in ("polygon", "circle_3p", "angle"):
            return
        elif self.current_tool == "circle":
            self.handle_circle_drag(start_pos, end_pos)
        elif self.current_tool == "caliper":
            self.handle_caliper(start_pos, end_pos)
        elif self.current_tool == "calibrate":
            self.handle_calibration(start_pos, end_pos)
        else:
            # 1. Map to real pixels first
            real_p1 = self.label.map_to_real(start_pos)
            real_p2 = self.label.map_to_real(end_pos)
            
            # 2. Calculate distance/size for validation
            dx = abs(real_p2[0] - real_p1[0])
            dy = abs(real_p2[1] - real_p1[1])
            dist = float(np.hypot(dx, dy))

            # 3. THRESHOLD CHECK
            # Ignore if length is less than 3px or area is 0
            if self.current_tool == "length" and dist < 3:
                print("Ignored: Length too short")
                return
            
            if self.current_tool in ("area", "square") and (dx < 2 or dy < 2):
                print("Ignored: Area too small")
                return

            tool_type = "square" if self.current_tool == "area" else self.current_tool
            # 4. If valid, create the measurement
            measurement = {
                'type': tool_type,
                'real_p1': real_p1,
                'real_p2': real_p2,
                'length': dist if tool_type == "length" else 0,
                'area': (dx * dy) if tool_type in ("square", "area") else 0
            }
            
            self.measurements.append(measurement)
            self.label.update()

    def handle_circle_drag(self, start_pos, end_pos):
        real_c = self.label.map_to_real(start_pos)
        real_e = self.label.map_to_real(end_pos)
        radius = float(np.hypot(real_e[0] - real_c[0], real_e[1] - real_c[1]))
        if radius < 3.0:
            print("Ignored: Circle radius too small")
            return
        measurement = {
            'type': 'circle',
            'center': real_c,
            'radius': radius,
            'diameter': radius * 2.0,
            'area': float(np.pi * (radius ** 2)),
            'circumference': float(2.0 * np.pi * radius),
            'method': 'drag'
        }
        self.measurements.append(measurement)
        self.label.update()

    def handle_caliper(self, start_pos, end_pos):
        if self.current_frame is None:
            print("Caliper: No active image frame to inspect.")
            return
        real_p1 = self.label.map_to_real(start_pos)
        real_p2 = self.label.map_to_real(end_pos)
        line_len = float(np.hypot(real_p2[0] - real_p1[0], real_p2[1] - real_p1[1]))
        if line_len < 4.0:
            print("Ignored: Caliper line too short")
            return

        from utils.metrology import run_1d_edge_caliper
        res = run_1d_edge_caliper(self.current_frame, real_p1, real_p2)
        if res.get('success') and res.get('edges'):
            edges_pos = [e['pos'] for e in res['edges']]
            measurement = {
                'type': 'caliper',
                'real_p1': real_p1,
                'real_p2': real_p2,
                'edges': edges_pos,
                'edge_details': res['edges'],
                'distance': res['distance'],
                'num_edges': len(res['edges'])
            }
            self.measurements.append(measurement)
            self.label.update()
        else:
            print("Caliper: No sharp edge transitions found along profile line.")

    def handle_calibration(self, start_pos, end_pos):
        real_p1 = self.label.map_to_real(start_pos)
        real_p2 = self.label.map_to_real(end_pos)
        dist = float(np.hypot(real_p2[0] - real_p1[0], real_p2[1] - real_p1[1]))
        if dist < 4.0:
            print("Ignored: Calibration line too short")
            return
        self.open_calibration_dialog(dist)

    def open_calibration_dialog(self, measured_px=None):
        if measured_px is None or measured_px < 2.0:
            ref_px = None
            for m in reversed(self.measurements):
                if m.get('type') == 'length' and m.get('length', 0) > 0:
                    ref_px = m['length']
                    break
                elif m.get('type') == 'caliper' and m.get('distance', 0) > 0:
                    ref_px = m['distance']
                    break
                elif m.get('type') == 'circle' and m.get('diameter', 0) > 0:
                    ref_px = m['diameter']
                    break
            if ref_px is not None:
                measured_px = ref_px
            else:
                QMessageBox.information(
                    self, "Scale Calibration",
                    "Please select 'Calibrate Scale' tool and drag a line across a feature of known real-world dimension (e.g. 10 mm ruler or gauge block)."
                )
                return

        from utils.calibration_dialog import CalibrationDialog
        dialog = CalibrationDialog(measured_px, self.calibration, self)
        if dialog.exec():
            data = dialog.get_calibration_data()
            apply_all = data.pop('apply_all', True)
            self.apply_calibration(data)
            if hasattr(self.main_window, 'global_calibration'):
                self.main_window.global_calibration = data.copy()
            if apply_all and hasattr(self.main_window, 'modules'):
                for mod in self.main_window.modules:
                    if mod != self:
                        mod.apply_calibration(data.copy())

    def apply_calibration(self, calib_data):
        self.calibration.update(calib_data)
        if hasattr(self, 'calib_val_label'):
            if self.calibration.get('is_calibrated', False):
                self.calib_val_label.setText(f"Scale: 1 px = {self.calibration['scale']:.4g} {self.calibration['unit']}")
                self.calib_val_label.show()
                if hasattr(self, 'calib_sep'):
                    self.calib_sep.show()
            else:
                self.calib_val_label.hide()
                if hasattr(self, 'calib_sep'):
                    self.calib_sep.hide()
        self.label.update()

    def reset_calibration(self):
        self.calibration = {
            'scale': 1.0,
            'unit': 'px',
            'is_calibrated': False,
            'ref_pixels': 0.0,
            'ref_real': 0.0
        }
        if hasattr(self, 'calib_val_label'):
            self.calib_val_label.hide()
        if hasattr(self, 'calib_sep'):
            self.calib_sep.hide()
        self.label.update()

    def format_length(self, px_len):
        if getattr(self, 'calibration', {}).get('is_calibrated', False):
            scale = self.calibration['scale']
            unit = self.calibration['unit']
            real_val = px_len * scale
            return f"{real_val:.2f} {unit} ({px_len:.1f} px)"
        return f"{px_len:.2f} px"

    def format_area(self, px_area):
        if getattr(self, 'calibration', {}).get('is_calibrated', False):
            scale = self.calibration['scale']
            unit = self.calibration['unit']
            real_val = px_area * (scale ** 2)
            return f"{real_val:.2f} {unit}² ({px_area:.0f} px²)"
        return f"{px_area:.2f} px²"

    def get_clean_title(self):
        """Returns the base module title without transient status suffixes"""
        raw = self.title_label.text() if hasattr(self, 'title_label') else f"Module {self.index}"
        raw = raw.replace("[Drag on video to Highlight Template]", "")
        raw = raw.replace("[Drag on video to Crop Region]", "")
        raw = raw.replace("[Click on video to Sample Color]", "").strip()
        import re
        raw = re.sub(r'\s+', ' ', raw)
        return raw or f"Module {self.index}"

    def update_index(self, new_index):
        """Updates the module's index and title label cleanly, preserving serial or custom title"""
        old_idx = self.index
        self.index = new_index
        if hasattr(self, 'title_label'):
            clean_title = self.get_clean_title()
            old_tag = f"Module {old_idx}"
            new_tag = f"Module {new_index}"
            if old_tag in clean_title:
                base_title = clean_title.replace(old_tag, new_tag, 1)
            elif clean_title.startswith("Module "):
                base_title = new_tag
            elif not clean_title or clean_title == old_tag:
                base_title = new_tag
            elif hasattr(self, 'serial_info') and self.serial_info and clean_title == str(self.serial_info):
                base_title = clean_title
            elif getattr(self, 'static_image', False) and hasattr(self, 'image_path') and self.image_path and Path(self.image_path).name == clean_title:
                base_title = clean_title
            else:
                base_title = new_tag

            suffix = ""
            if getattr(self, 'current_tool', None) == "template":
                suffix = " [Drag on video to Highlight Template]"
            elif getattr(self, 'current_tool', None) == "crop_drag":
                suffix = " [Drag on video to Crop Region]"
            elif getattr(self, 'current_tool', None) == "pick_color":
                suffix = " [Click on video to Sample Color]"

            self.title_label.setText(f"{base_title}{suffix}")

    def start_template_highlight(self, callback=None, on_cancel=None):
        """Toggles template highlight mode allowing operator to drag an ROI on the video (ON/OFF mode)"""
        if getattr(self, 'current_tool', None) == "template":
            self.stop_template_highlight()
            return False

        self._template_callback = callback
        self._template_cancel_callback = on_cancel
        prev = getattr(self, 'current_tool', 'zoom')
        self._previous_tool = prev if prev != "template" else "zoom"
        self.current_tool = "template"

        clean_title = self.get_clean_title()
        self._orig_title = clean_title
        if hasattr(self, 'title_label'):
            self.title_label.setText(f"{clean_title} [Drag on video to Highlight Template]")
        print(f"Module {self.index}: Template highlight mode activated. Drag on live video.")
        return True

    def stop_template_highlight(self):
        """Exits template highlight mode and restores previous tool and title"""
        clean_title = self.get_clean_title()
        if hasattr(self, 'title_label'):
            self.title_label.setText(clean_title)
        
        cancel_cb = getattr(self, '_template_cancel_callback', None)
        if cancel_cb and callable(cancel_cb):
            try:
                cancel_cb()
            except Exception:
                pass
        self._template_cancel_callback = None
        self._template_callback = None

        self.current_tool = getattr(self, '_previous_tool', 'zoom')
        if self.current_tool == "template":
            self.current_tool = "zoom"
        if hasattr(self, 'label') and hasattr(self.label, 'update'):
            self.label.update()
        print(f"Module {self.index}: Template highlight mode deactivated.")

    def start_color_pick(self, callback=None, on_cancel=None):
        """Toggles single-click color sampling mode from live video (ON/OFF mode)"""
        if getattr(self, 'current_tool', None) == "pick_color":
            self.stop_color_pick()
            return False

        self._color_pick_callback = callback
        self._color_pick_cancel_callback = on_cancel
        prev = getattr(self, 'current_tool', 'zoom')
        self._previous_tool = prev if prev != "pick_color" else "zoom"
        self.current_tool = "pick_color"

        clean_title = self.get_clean_title()
        self._orig_title = clean_title
        if hasattr(self, 'title_label'):
            self.title_label.setText(f"{clean_title} [Click on video to Sample Color]")
        print(f"Module {self.index}: Click on live video to sample target pixel color.")
        return True

    def stop_color_pick(self):
        """Exits color pick mode and restores previous tool and title"""
        clean_title = self.get_clean_title()
        if hasattr(self, 'title_label'):
            self.title_label.setText(clean_title)

        cancel_cb = getattr(self, '_color_pick_cancel_callback', None)
        if cancel_cb and callable(cancel_cb):
            try:
                cancel_cb()
            except Exception:
                pass
        self._color_pick_cancel_callback = None
        self._color_pick_callback = None

        self.current_tool = getattr(self, '_previous_tool', 'zoom')
        if self.current_tool == "pick_color":
            self.current_tool = "zoom"
        if hasattr(self, 'label') and hasattr(self.label, 'update'):
            self.label.update()
        print(f"Module {self.index}: Color pick mode deactivated.")

    def start_crop_drag(self, callback=None, on_cancel=None):
        """Toggles crop ROI drag mode on the video (ON/OFF mode)"""
        if getattr(self, 'current_tool', None) == "crop_drag":
            self.stop_crop_drag()
            return False

        self._crop_callback = callback
        self._crop_cancel_callback = on_cancel
        prev = getattr(self, 'current_tool', 'zoom')
        self._previous_tool = prev if prev != "crop_drag" else "zoom"
        self.current_tool = "crop_drag"

        clean_title = self.get_clean_title()
        self._orig_title = clean_title
        if hasattr(self, 'title_label'):
            self.title_label.setText(f"{clean_title} [Drag on video to Crop Region]")
        print(f"Module {self.index}: Crop drag mode activated. Drag on live video.")
        return True

    def stop_crop_drag(self):
        """Exits crop drag mode and restores previous tool and title"""
        clean_title = self.get_clean_title()
        if hasattr(self, 'title_label'):
            self.title_label.setText(clean_title)

        cancel_cb = getattr(self, '_crop_cancel_callback', None)
        if cancel_cb and callable(cancel_cb):
            try:
                cancel_cb()
            except Exception:
                pass
        self._crop_cancel_callback = None
        self._crop_callback = None

        self.current_tool = getattr(self, '_previous_tool', 'zoom')
        if self.current_tool == "crop_drag":
            self.current_tool = "zoom"
        if hasattr(self, 'label') and hasattr(self.label, 'update'):
            self.label.update()
        print(f"Module {self.index}: Crop drag mode deactivated.")

    def capture_crop_from_roi(self, start_pos, end_pos):
        """Calculates cropped bounding box coordinates (x, y, w, h) from dragged ROI"""
        real_p1 = self.label.map_to_real(start_pos)
        real_p2 = self.label.map_to_real(end_pos)

        x1 = min(real_p1[0], real_p2[0])
        x2 = max(real_p1[0], real_p2[0])
        y1 = min(real_p1[1], real_p2[1])
        y2 = max(real_p1[1], real_p2[1])

        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)

        # Update active pipeline node if Crop exists in process canvas
        if hasattr(self, 'process_window') and self.process_window and hasattr(self.process_window, 'canvas'):
            canvas = self.process_window.canvas
            for item in canvas.scene.items():
                if hasattr(item, 'title') and item.title == "Crop":
                    item.parameters["x_pos"] = int(x1)
                    item.parameters["y_pos"] = int(y1)
                    item.parameters["width"] = int(crop_w)
                    item.parameters["height"] = int(crop_h)
                    item.parameters["mode"] = "Drag"
                    if hasattr(canvas, 'push_live_pipeline'):
                        canvas.push_live_pipeline()

        cb = getattr(self, '_crop_callback', None)
        if cb and callable(cb):
            cb(int(x1), int(y1), int(crop_w), int(crop_h))
            self._crop_callback = None

        self.stop_crop_drag()
        print(f"Module {self.index}: Crop region selected: x={x1}, y={y1}, w={crop_w}, h={crop_h}")
        self.label.update()

    def capture_template_from_roi(self, start_pos, end_pos):
        """Crops selected region from current frame and sets it as the active template"""
        if self.current_frame is None:
            print("Cannot capture template: no active video frame.")
            self.stop_template_highlight()
            return

        real_p1 = self.label.map_to_real(start_pos)
        real_p2 = self.label.map_to_real(end_pos)

        x1 = min(real_p1[0], real_p2[0])
        x2 = max(real_p1[0], real_p2[0])
        y1 = min(real_p1[1], real_p2[1])
        y2 = max(real_p1[1], real_p2[1])

        h, w = self.current_frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < 5 or crop_h < 5:
            print("Ignored: Template selection area too small (minimum 5x5 px).")
            self.stop_template_highlight()
            return

        tpl_patch = self.current_frame[y1:y2, x1:x2].copy()

        from pathlib import Path
        save_dir = getattr(self.main_window, 'dir_path', None)
        if save_dir:
            tpl_dir = Path(save_dir) / "templates"
        else:
            tpl_dir = Path("./media_output/templates")
        tpl_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tpl_path = str((tpl_dir / f"template_{timestamp}.png").resolve())
        cv2.imwrite(tpl_path, tpl_patch)

        # Update in-memory live template in computer_vision.template_matching
        try:
            import computer_vision.template_matching as tm
            tm.set_live_template(tpl_patch, tpl_path)
        except Exception as e:
            print(f"Failed setting live template: {e}")

        # Update active pipeline node if Template Matching exists
        if hasattr(self, 'process_window') and self.process_window and hasattr(self.process_window, 'canvas'):
            canvas = self.process_window.canvas
            for item in canvas.scene.items():
                if hasattr(item, 'title') and item.title == "Template Matching":
                    item.parameters["template_file"] = tpl_path
                    if hasattr(canvas, 'push_live_pipeline'):
                        canvas.push_live_pipeline()

        # Invoke callback if dialog was waiting
        cb = getattr(self, '_template_callback', None)
        if cb and callable(cb):
            cb(tpl_path, tpl_patch)
            self._template_callback = None

        # Restore tool and clean title
        self.stop_template_highlight()

        print(f"Module {self.index}: Template captured ({crop_w}x{crop_h} px) -> {tpl_path}")
        self.label.update()

    def apply_zoom(self, start_pos, end_pos):
        """Maps the drag-box coordinates to the real image resolution."""
        if self.current_frame is None or not self.label.pixmap() or self.label.pixmap().isNull():
            return
        
        min_zoom_size = 20

        # 1. Get geometry of the displayed image inside the label
        pix_size = self.label.pixmap().size()
        lbl_size = self.label.size()
        x_off = (lbl_size.width() - pix_size.width()) // 2
        y_off = (lbl_size.height() - pix_size.height()) // 2

        # 2. Map screen coordinates to the current view's pixels
        rect = QRect(start_pos, end_pos).normalized()
        
        # Scale factor between displayed pixmap and current view resolution
        if self.zoom_rect:
            curr_zx, curr_zy, view_w, view_h = self.zoom_rect
        else:
            curr_zx, curr_zy = 0, 0
            view_w, view_h = self.label.real_resolution
            if view_w <= 0 or view_h <= 0:
                view_w, view_h = self.current_frame.shape[1], self.current_frame.shape[0]

        scale_x = view_w / pix_size.width() if pix_size.width() > 0 else 1.0
        scale_y = view_h / pix_size.height() if pix_size.height() > 0 else 1.0

        # Local image coordinates relative to visible area
        lx = max(0, int((rect.left() - x_off) * scale_x))
        ly = max(0, int((rect.top() - y_off) * scale_y))
        lw = int(rect.width() * scale_x)
        lh = int(rect.height() * scale_y)

        # Clamp width and height inside current view
        lw = min(lw, max(0, view_w - lx))
        lh = min(lh, max(0, view_h - ly))

        # 3. Apply to global image coordinates
        if lw > min_zoom_size and lh > min_zoom_size:
            self.zoom_rect = (curr_zx + lx, curr_zy + ly, lw, lh)
        
        self.label.update()

    def capture_image(self, save_dir):
        if self.current_frame is not None:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"module_{self.index}_{timestamp}.png")
            success = cv2.imwrite(save_path, self.current_frame)
            if success:
                print(f"Module {self.index} image saved at {save_path}")
                if hasattr(self.main_window, 'dir_viewer') and self.main_window.dir_viewer:
                    self.main_window.dir_viewer.update_file_list()
            else:
                print(f"Failed to write image to {save_path}")
        else:
            print("No active frame to capture!")

    # def capture_image(self, save_dir):
    #     pixmap = self.label.grab()
        
    #     if not pixmap.isNull():
    #         save_path = os.path.join(save_dir, f"capture_{int(time.time())}.png")
    #         pixmap.save(save_path, "PNG")
    #         print(f"Captured screen view to {save_path}")
    #     else:
    #         print("Capture failed: No pixmap available")

    def record_video(self, save_dir):
        if self.current_frame is None:
            print("No active frame to record!")
            return

        # Stop any previous recording just in case
        self.stop_video()

        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(save_dir, f"module_{self.index}_{timestamp}.avi")

        self.record_height, self.record_width = self.current_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            self.video_path, fourcc, 20.0, (self.record_width, self.record_height), isColor=True
        )

        if not self.video_writer.isOpened():
            print(f"Failed to open VideoWriter for {self.video_path}")
            self.video_writer = None
            self.is_recording = False
            return

        self.is_recording = True
        print(f"Module {self.index}: Started recording at {self.video_path}")

    def stop_video(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print(f"Module {self.index}: Stopped recording. Saved to {self.video_path}")
            if hasattr(self.main_window, 'dir_viewer') and self.main_window.dir_viewer:
                self.main_window.dir_viewer.update_file_list()
            self.video_path = None

    def open_process_editor(self):
        if self.process_window is None:
            self.process_window = ProcessWindow(self)
            if self.saved_nodes:
                self.process_window.canvas.restore_setup(self.saved_nodes)
            else:
                self.process_window.send_to_worker()
        
        if hasattr(self.main_window, 'show_module_process_editor'):
            self.main_window.show_module_process_editor(self, force_show=True)
        else:
            self.process_window.show()
            self.process_window.raise_()
            self.process_window.activateWindow()

    def request_close(self):
        if getattr(self, 'is_fullscreen_window', False):
            self.exit_fullscreen_window()
        self.close_event_logic()
        self.remove_from_active_cameras()
        gui_signal.closed_module.emit(self)

    def close_event_logic(self):
        """Safely stops and cleans up the thread and associated resources."""
        # Stop recording if active
        if self.is_recording:
            self.stop_video()

        if self.worker:
            try:
                self.worker.image_ready.disconnect(self.update_view)
            except Exception:
                pass
            
        if self.thread and self.thread.isRunning():
            self.worker.stop()     # Tell the worker loop to break
            self.thread.quit()     # Tell the thread to stop
            self.thread.wait(2000) # Wait to actually finish

        # Now that worker thread is stopped, safely close camera object
        if self.worker and hasattr(self.worker, 'camera_object') and not self.static_image:
            try:
                self.worker.camera_object.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

        # Clean up camera parameter panel if present
        if hasattr(self, 'camera_panel') and self.camera_panel:
            if hasattr(self.main_window, 'right_panel'):
                self.main_window.right_panel.removeWidget(self.camera_panel)
            self.camera_panel.deleteLater()
            self.camera_panel = None

        if self.process_window:
            # Save the current state before closing/destroying
            self.saved_nodes = self.process_window.canvas.get_save_data()
            if hasattr(self.main_window, 'process_panel') and self.main_window.process_panel:
                self.main_window.process_panel.removeWidget(self.process_window)
            self.process_window.close()
            self.process_window.deleteLater()
            self.process_window = None

        self.static_image = False
        
        self.thread = None
        self.worker = None
        self.current_frame = None

    def remove_from_active_cameras(self):
        if hasattr(self, "serial_info") and self.serial_info in self.main_window.active_cameras:
            self.main_window.active_cameras.remove(self.serial_info)