# scripts to generate parameter dialog for computer vision modules
from PySide6.QtWidgets import (QDialog, QFormLayout, QCheckBox, QSpinBox,
                               QDoubleSpinBox, QPushButton, QComboBox,
                               QWidget, QHBoxLayout, QLineEdit, QFileDialog,
                               QColorDialog, QDialogButtonBox)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from .deep_learning import get_ai_scripts

class ParameterDialog(QDialog):
    parametersChanged = Signal(dict)
    
    def __init__(self, title, params_def, current_values, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(f"{title} Parameters")
        self.setModal(False)
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.layout = QFormLayout(self)
        self.setMinimumWidth(350)
        self.widgets = {}
        self.params_def = params_def

        for name, config in params_def.items():
            label_text = name.replace("_", " ").title()
            current_val = current_values.get(name, config.get("default"))

            if config["type"] == "enum":
                w = QComboBox()
                w.addItems(config["options"])
                
                w.setCurrentText(str(current_val))
                
                w.currentTextChanged.connect(self.emit_updates)
                
                if name == "mode" and ("width" in self.params_def or "x_pos" in self.params_def) and "Drag" in config.get("options", []):
                    container = QWidget()
                    h_layout = QHBoxLayout(container)
                    h_layout.setContentsMargins(0, 0, 0, 0)
                    h_layout.setSpacing(6)
                    h_layout.addWidget(w)

                    btn_drag_crop = QPushButton("Drag Crop on Video")
                    btn_drag_crop.setToolTip("Click and drag a box on the live video module to select crop region (click again to cancel)")
                    btn_drag_crop.setStyleSheet("background-color: #0078d4; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")
                    btn_drag_crop.setCursor(Qt.PointingHandCursor)

                    def reset_crop_btn(btn=btn_drag_crop):
                        btn.setText("Drag Crop on Video")
                        btn.setStyleSheet("background-color: #0078d4; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")

                    def trigger_crop_drag(b_cr=btn_drag_crop):
                        mod = self.find_window_module()
                        if mod and hasattr(mod, 'start_crop_drag'):
                            if getattr(mod, 'current_tool', None) == "crop_drag":
                                mod.stop_crop_drag()
                                reset_crop_btn(b_cr)
                                return

                            b_cr.setText("Cancel Crop Drag")
                            b_cr.setStyleSheet("background-color: #d83b01; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")

                            def on_crop_selected(rx, ry, rw, rh):
                                if "x_pos" in self.widgets:
                                    self.widgets["x_pos"].setValue(rx)
                                if "y_pos" in self.widgets:
                                    self.widgets["y_pos"].setValue(ry)
                                if "width" in self.widgets:
                                    self.widgets["width"].setValue(rw)
                                if "height" in self.widgets:
                                    self.widgets["height"].setValue(rh)
                                reset_crop_btn(b_cr)
                                self.emit_updates()

                            def on_cancelled():
                                reset_crop_btn(b_cr)

                            mod.start_crop_drag(callback=on_crop_selected, on_cancel=on_cancelled)

                    btn_drag_crop.clicked.connect(lambda _, bcr=btn_drag_crop: trigger_crop_drag(bcr))
                    h_layout.addWidget(btn_drag_crop)

                    def update_crop_visibility(val, btn=btn_drag_crop):
                        is_drag = (val == "Drag")
                        btn.setVisible(is_drag)
                        if is_drag:
                            # Reset crop parameters so user sees full image and can drag an ROI
                            if "x_pos" in self.widgets:
                                self.widgets["x_pos"].setValue(0)
                            if "y_pos" in self.widgets:
                                self.widgets["y_pos"].setValue(0)
                            if "width" in self.widgets:
                                self.widgets["width"].setValue(0)
                            if "height" in self.widgets:
                                self.widgets["height"].setValue(0)
                            self.emit_updates()
                        else:
                            # Restoring to Input mode: restore default input size if zero
                            if "width" in self.widgets and self.widgets["width"].value() == 0:
                                self.widgets["width"].setValue(640)
                            if "height" in self.widgets and self.widgets["height"].value() == 0:
                                self.widgets["height"].setValue(480)
                            mod = self.find_window_module()
                            if mod and getattr(mod, 'current_tool', None) == "crop_drag":
                                mod.stop_crop_drag()
                            reset_crop_btn(btn)
                            self.emit_updates()

                    w.currentTextChanged.connect(update_crop_visibility)
                    update_crop_visibility(w.currentText())

                    self.layout.addRow(label_text, container)
                    self.widgets[name] = w
                else:
                    self.layout.addRow(label_text, w)
                    self.widgets[name] = w

            elif config["type"] == "ai_modules":
                w = QComboBox()
                w.addItems(get_ai_scripts())
                
                w.setCurrentText(str(current_val))
                
                w.currentTextChanged.connect(self.emit_updates)
                
                self.layout.addRow(label_text, w)
                self.widgets[name] = w

            elif config["type"] == "int":
                spin = QSpinBox()
                spin.setRange(config.get("min", 0), config.get("max", 100))
                
                spin.setSingleStep(config.get("step", 1))
                
                spin.setValue(int(current_val))
                spin.valueChanged.connect(self.emit_updates)
                self.layout.addRow(label_text, spin)
                self.widgets[name] = spin
                
            elif config["type"] == "float":
                f_spin = QDoubleSpinBox()
                decimals = config.get("decimals", 2)
                f_spin.setDecimals(decimals)
                f_spin.setRange(config.get("min", 0.0), config.get("max", 100.0))
                f_spin.setSingleStep(config.get("step", 0.1))
                f_spin.setValue(float(current_val))
                f_spin.valueChanged.connect(self.emit_updates)
                self.layout.addRow(label_text, f_spin)
                self.widgets[name] = f_spin

            elif config["type"] == "bool":
                check = QCheckBox()
                check.setChecked(bool(current_val))
                check.toggled.connect(self.emit_updates)
                self.layout.addRow(label_text, check)
                self.widgets[name] = check

            elif config["type"] == "color":
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                h_layout.setSpacing(6)

                initial_color_hex = str(current_val or "#00ff00")
                line_edit = QLineEdit(initial_color_hex)
                line_edit.setMinimumWidth(80)

                btn_color = QPushButton()
                btn_color.setFixedSize(36, 24)
                btn_color.setCursor(Qt.PointingHandCursor)
                btn_color.setToolTip("Click to open color picker")

                def update_swatch(hex_code, btn=btn_color):
                    if QColor.isValidColor(hex_code):
                        btn.setStyleSheet(
                            f"background-color: {hex_code}; border: 1px solid #7a7a7a; border-radius: 3px;"
                        )

                update_swatch(initial_color_hex)

                def pick_color(le=line_edit, btn=btn_color):
                    curr_qcolor = QColor(le.text()) if QColor.isValidColor(le.text()) else QColor("#00ff00")
                    chosen = QColorDialog.getColor(curr_qcolor, self, "Pick Target Color")
                    if chosen.isValid():
                        hex_val = chosen.name()
                        le.setText(hex_val)
                        update_swatch(hex_val, btn)

                btn_color.clicked.connect(lambda _, le=line_edit, btn=btn_color: pick_color(le, btn))
                line_edit.textChanged.connect(lambda text, btn=btn_color: update_swatch(text, btn))
                line_edit.textChanged.connect(self.emit_updates)

                btn_pick_video = QPushButton("Pick from Video")
                btn_pick_video.setToolTip("Click anywhere on the live video module to sample that pixel's color (click again to cancel)")
                btn_pick_video.setStyleSheet("background-color: #0078d4; color: white; padding: 2px 8px; border-radius: 3px;")
                btn_pick_video.setCursor(Qt.PointingHandCursor)

                def reset_pick_btn(btn=btn_pick_video):
                    btn.setText("Pick from Video")
                    btn.setStyleSheet("background-color: #0078d4; color: white; padding: 2px 8px; border-radius: 3px;")

                def trigger_video_color_pick(le=line_edit, btn_sw=btn_color, b_pk=btn_pick_video):
                    mod = self.find_window_module()
                    if mod and hasattr(mod, 'start_color_pick'):
                        if getattr(mod, 'current_tool', None) == "pick_color":
                            mod.stop_color_pick()
                            reset_pick_btn(b_pk)
                            return

                        b_pk.setText("Cancel Pick")
                        b_pk.setStyleSheet("background-color: #d83b01; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")

                        def on_color_chosen(hex_c):
                            le.setText(hex_c)
                            update_swatch(hex_c, btn_sw)
                            reset_pick_btn(b_pk)
                            self.emit_updates()

                        def on_cancelled():
                            reset_pick_btn(b_pk)

                        mod.start_color_pick(callback=on_color_chosen, on_cancel=on_cancelled)

                btn_pick_video.clicked.connect(lambda _, le=line_edit, btn=btn_color, bpk=btn_pick_video: trigger_video_color_pick(le, btn, bpk))

                h_layout.addWidget(btn_color)
                h_layout.addWidget(line_edit)
                h_layout.addWidget(btn_pick_video)

                self.layout.addRow(label_text, container)
                self.widgets[name] = line_edit

            elif config["type"] == "file":
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                h_layout.setSpacing(6)
                
                line_edit = QLineEdit(str(current_val or ""))
                btn_browse = QPushButton("...")
                btn_browse.setFixedWidth(30)
                btn_browse.setToolTip("Browse template image file from disk")
                
                h_layout.addWidget(line_edit)
                h_layout.addWidget(btn_browse)
                
                file_filter = config.get("filter", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*.*)")
                
                def browse_file(le, f_filter):
                    file_path, _ = QFileDialog.getOpenFileName(
                        self, 
                        "Select File",
                        le.text() or "",
                        f_filter
                    )
                    if file_path:
                        le.setText(file_path)

                btn_browse.clicked.connect(lambda _, le=line_edit, ff=file_filter: browse_file(le, ff))

                if config.get("allow_live_capture", False):
                    import os
                    from PySide6.QtGui import QPixmap
                    from PySide6.QtWidgets import QLabel

                    btn_grab = QPushButton("Highlight on Video")
                    btn_grab.setToolTip("Click and drag a box on the live video module to highlight and crop a template (click again to cancel)")
                    btn_grab.setStyleSheet("background-color: #0078d4; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")
                    btn_grab.setCursor(Qt.PointingHandCursor)

                    def reset_grab_btn(btn=btn_grab):
                        btn.setText("Highlight on Video")
                        btn.setStyleSheet("background-color: #0078d4; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")

                    lbl_preview = QLabel()
                    lbl_preview.setFixedSize(32, 24)
                    lbl_preview.setStyleSheet("border: 1px solid #777; background: #222; border-radius: 2px;")
                    lbl_preview.setScaledContents(True)

                    def update_thumb(path_val, lbl=lbl_preview):
                        if path_val and os.path.isfile(path_val):
                            pm = QPixmap(path_val)
                            if not pm.isNull():
                                lbl.setPixmap(pm)
                                lbl.show()
                                return
                        lbl.clear()

                    update_thumb(str(current_val or ""))

                    def trigger_live_grab(le=line_edit, lbl=lbl_preview, b_gb=btn_grab):
                        mod = self.find_window_module()
                        if mod and hasattr(mod, 'start_template_highlight'):
                            if getattr(mod, 'current_tool', None) == "template":
                                mod.stop_template_highlight()
                                reset_grab_btn(b_gb)
                                return

                            b_gb.setText("Cancel Highlight")
                            b_gb.setStyleSheet("background-color: #d83b01; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;")

                            def on_captured(tpl_path, _tpl_img):
                                le.setText(tpl_path)
                                update_thumb(tpl_path, lbl)
                                reset_grab_btn(b_gb)
                                self.emit_updates()

                            def on_cancelled():
                                reset_grab_btn(b_gb)

                            mod.start_template_highlight(callback=on_captured, on_cancel=on_cancelled)

                    btn_grab.clicked.connect(lambda _, le=line_edit, lbl=lbl_preview, bgb=btn_grab: trigger_live_grab(le, lbl, bgb))
                    line_edit.textChanged.connect(lambda txt, lbl=lbl_preview: update_thumb(txt, lbl))

                    h_layout.addWidget(btn_grab)
                    h_layout.addWidget(lbl_preview)

                    if "source" in self.params_def:
                        def update_source_controls(src_val, b_br=btn_browse, b_gb=btn_grab, lbl_p=lbl_preview):
                            if src_val == "Drag":
                                b_br.setVisible(False)
                                b_gb.setVisible(True)
                                lbl_p.setVisible(True)
                            else:
                                b_br.setVisible(True)
                                b_gb.setVisible(False)
                                lbl_p.setVisible(False)
                                mod = self.find_window_module()
                                if mod and getattr(mod, 'current_tool', None) == "template":
                                    mod.stop_template_highlight()
                                reset_grab_btn(b_gb)

                        self._update_source_visibility = update_source_controls

                line_edit.textChanged.connect(self.emit_updates)
                
                self.layout.addRow(label_text, container)
                self.widgets[name] = line_edit

            elif config["type"] == "dir":
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                
                line_edit = QLineEdit(str(current_val or ""))
                btn_browse = QPushButton("...")
                btn_browse.setFixedWidth(30)
                
                h_layout.addWidget(line_edit)
                h_layout.addWidget(btn_browse)
                
                # define the browse callback
                def browse(le, n):
                    dir_path = QFileDialog.getExistingDirectory(
                        self, 
                        "Select Folder",
                        le.text() or ""
                    )
                    if dir_path:
                        le.setText(dir_path)

                btn_browse.clicked.connect(lambda _, le=line_edit, n=name: browse(le, n))
                line_edit.textChanged.connect(self.emit_updates)
                
                self.layout.addRow(label_text, container)
                self.widgets[name] = line_edit

        # Wire dynamic cross-parameter dependencies if present
        if hasattr(self, '_update_source_visibility') and "source" in self.widgets:
            source_w = self.widgets["source"]
            if isinstance(source_w, QComboBox):
                source_w.currentTextChanged.connect(self._update_source_visibility)
                self._update_source_visibility(source_w.currentText())

        if "global_tolerance" in self.widgets:
            gt_spin = self.widgets["global_tolerance"]
            def sync_tolerances(val):
                if "tolerance_hue" in self.widgets:
                    self.widgets["tolerance_hue"].setValue(max(1, min(90, int(round(90 * (val / 100.0))))))
                if "tolerance_sat" in self.widgets:
                    self.widgets["tolerance_sat"].setValue(max(1, min(255, int(round(255 * (val / 100.0))))))
                if "tolerance_val" in self.widgets:
                    self.widgets["tolerance_val"].setValue(max(1, min(255, int(round(255 * (val / 100.0))))))
                self.emit_updates()
            gt_spin.valueChanged.connect(sync_tolerances)

        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.layout.addRow(self.button_box)

    def _cleanup_live_tools(self):
        mod = self.find_window_module()
        if mod:
            if getattr(mod, 'current_tool', None) == "template" and hasattr(mod, 'stop_template_highlight'):
                mod.stop_template_highlight()
            if getattr(mod, 'current_tool', None) == "crop_drag" and hasattr(mod, 'stop_crop_drag'):
                mod.stop_crop_drag()
            if getattr(mod, 'current_tool', None) == "pick_color" and hasattr(mod, 'stop_color_pick'):
                mod.stop_color_pick()

    def closeEvent(self, event):
        self._cleanup_live_tools()
        super().closeEvent(event)

    def reject(self):
        self._cleanup_live_tools()
        super().reject()

    def accept(self):
        self._cleanup_live_tools()
        super().accept()

    def get_values(self):
        result = {}
        for name, widget in self.widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                result[name] = widget.value()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                result[name] = widget.text()
        return result

    def emit_updates(self):
        """Emits the current state of all widgets immediately"""
        self.parametersChanged.emit(self.get_values())

    def find_window_module(self):
        """Locates the active WindowModule from parent tree or application windows"""
        curr = self.parent()
        while curr is not None:
            if hasattr(curr, 'window_module') and curr.window_module is not None:
                return curr.window_module
            if hasattr(curr, 'active_module') and curr.active_module is not None:
                return curr.active_module
            if hasattr(curr, 'main_window') and hasattr(curr.main_window, 'active_module'):
                return curr.main_window.active_module
            if hasattr(curr, 'modules') and len(curr.modules) > 0:
                return getattr(curr, 'active_module', None) or curr.modules[0]
            curr = curr.parent() if hasattr(curr, 'parent') and callable(curr.parent) else None

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            for top in app.topLevelWidgets():
                if hasattr(top, 'active_module') and top.active_module is not None:
                    return top.active_module
                if hasattr(top, 'modules') and len(top.modules) > 0:
                    return getattr(top, 'active_module', None) or top.modules[0]
        return None