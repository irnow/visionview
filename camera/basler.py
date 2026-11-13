from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QFormLayout,
    QComboBox, QTabWidget, QScrollArea, QTreeWidget,
    QTreeWidgetItem, QLineEdit, QPushButton, QHeaderView,
    QFrame
)
from PySide6.QtCore import Qt, QThread, QTimer

from pypylon import pylon, genicam
from .camera_parameters import BASLER_PARAMETERS

BASIC_TO_GENICAM = {
    "pixel format": "PixelFormat",
    "exposure": "ExposureTime",
    "gain": "Gain",
    "contrast": "BslContrast",
    "hue": "BslHue",
    "saturation": "BslSaturation",
    "brightness": "BslBrightness",
    "reversex": "ReverseX",
    "reversey": "ReverseY"
}

GENICAM_TO_BASIC = {v: k for k, v in BASIC_TO_GENICAM.items()}


class Basler():
    def __init__(self):
        print("Camera: Basler")
        self.camera = None
        self.converter = None
        self._active_parameters = {
            name: cfg["default"]
            for name, cfg in BASLER_PARAMETERS.items()
        }

        self._tab_widget = None
        self._basic_controls = {}
        self._advanced_controls = {}
        self._advanced_tree = None
        self._search_bar = None
        self._editable_chk = None

    def find(self):
        '''List of device instances and their names (or serial number) respectively.'''
        try:
            devices = list(pylon.TlFactory.GetInstance().EnumerateDevices())
            sn = [str(device.GetSerialNumber()) for device in devices]
            return devices, sn
        except Exception as e:
            print(f"Error enumerating Basler devices: {e}")
            return [], []

    def setup(self, device):
        converter = pylon.ImageFormatConverter()
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.converter = converter
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        self.camera.Open()

        try:
            self.camera.UserSetSelector.Value = "Default"
            self.camera.UserSetLoad.Execute()
        except Exception:
            pass

        try:
            self.camera.PixelFormat.Value = "BayerGB8"
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        except Exception:
            pass

        try:
            self.camera.BslLightSourcePreset.Value = "Daylight6500K"
        except Exception:
            pass

        try:
            self.camera.ExposureAuto.Value = "Off"
            self.camera.GainAuto.Value = "Off"
        except Exception:
            pass

        try:
            self.camera.BalanceWhiteAuto.Value = "Continuous"
        except Exception:
            pass

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def image(self):
        if self.camera is None:
            return None

        if not self.camera.IsGrabbing():
            return None

        grabResult = None

        try:
            grabResult = self.camera.RetrieveResult(
                1000, pylon.TimeoutHandling_Return
            )

            if grabResult and grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                return img

        except Exception as e:
            print("Grab error:", e)

        finally:
            if grabResult:
                grabResult.Release()

        return None

    def close(self):
        if self.camera is None:
            return

        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            if self.camera.IsOpen():
                self.camera.Close()

        except Exception as e:
            print("Close error:", e)

        self.camera = None
        self._advanced_controls.clear()
        self._basic_controls.clear()
        self._advanced_tree = None
        self._tab_widget = None

    # ------------------------------------------------
    # PARAMETER UI (Basic | Advanced)
    # ------------------------------------------------
    def parameter_widget(self):
        tab_widget = QTabWidget()
        self._tab_widget = tab_widget

        tab_widget.setStyleSheet("""
            QTabWidget {
                background-color: palette(window);
            }
            QTabWidget::pane {
                background-color: palette(window);
                border: none;
            }
            QTabBar::tab {
                background-color: #e4e7eb;
                color: #333333;
                padding: 6px 16px;
                margin-right: 2px;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: palette(window);
                color: #0078d4;
                font-weight: bold;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover:!selected {
                background-color: #d8dce2;
            }
            QScrollArea {
                background-color: palette(window);
                border: none;
            }
            QWidget#basic_inner, QWidget#adv_widget {
                background-color: palette(window);
            }
            QLabel {
                background: transparent;
                color: #000000;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 3px 22px 3px 6px;
                color: #000000;
                min-height: 24px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #888888;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #0078d4;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #dcdcdc;
                border-top-right-radius: 4px;
                background-color: #f8f8f8;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #e5e5e5;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background-color: #d0d0d0;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url("icons/arrow_up.png");
                width: 9px;
                height: 6px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border-left: 1px solid #dcdcdc;
                border-bottom-right-radius: 4px;
                background-color: #f8f8f8;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #e5e5e5;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #d0d0d0;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url("icons/arrow_down.png");
                width: 9px;
                height: 6px;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 3px 22px 3px 6px;
                color: #000000;
                min-height: 24px;
            }
            QComboBox:hover {
                border: 1px solid #888888;
            }
            QComboBox:focus {
                border: 1px solid #0078d4;
            }
            QComboBox::drop-down {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #dcdcdc;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background-color: #f8f8f8;
            }
            QComboBox::drop-down:hover {
                background-color: #e5e5e5;
            }
            QComboBox::down-arrow {
                image: url("icons/arrow_down.png");
                width: 9px;
                height: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #000000;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 2px;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 3px 6px;
                color: #000000;
                min-height: 24px;
            }
            QLineEdit:hover {
                border: 1px solid #888888;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator,
            QTreeWidget::indicator,
            QTreeView::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #7a7a7a;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:hover,
            QTreeWidget::indicator:hover,
            QTreeView::indicator:hover {
                border: 1px solid #0078d4;
            }
            QCheckBox::indicator:checked,
            QTreeWidget::indicator:checked,
            QTreeView::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                image: none;
            }
            QCheckBox::indicator:checked:hover,
            QTreeWidget::indicator:checked:hover,
            QTreeView::indicator:checked:hover {
                background-color: #106ebe;
                border: 1px solid #106ebe;
                image: none;
            }
            QCheckBox::indicator:checked:pressed,
            QTreeWidget::indicator:checked:pressed,
            QTreeView::indicator:checked:pressed {
                background-color: #005a9e;
                border: 1px solid #005a9e;
                image: none;
            }
            QCheckBox::indicator:disabled,
            QTreeWidget::indicator:disabled,
            QTreeView::indicator:disabled {
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
            }
            QCheckBox::indicator:checked:disabled,
            QTreeWidget::indicator:checked:disabled,
            QTreeView::indicator:checked:disabled {
                background-color: #a6c8e0;
                border: 1px solid #a6c8e0;
                image: none;
            }
        """)

        # -----------------
        # 1. Basic Tab
        # -----------------
        basic_scroll = QScrollArea()
        basic_scroll.setWidgetResizable(True)
        basic_scroll.setFrameShape(QFrame.NoFrame)

        basic_inner = QWidget()
        basic_inner.setObjectName("basic_inner")
        basic_layout = QFormLayout(basic_inner)
        basic_layout.setContentsMargins(10, 10, 10, 10)
        basic_layout.setSpacing(8)

        self._basic_controls.clear()

        for name, cfg in BASLER_PARAMETERS.items():
            param_type = cfg["type"]

            if param_type == "enum":
                w = QComboBox()
                w.addItems(cfg["options"])
                w.setCurrentText(str(self._active_parameters.get(name, cfg.get("default"))))
                w.currentTextChanged.connect(
                    lambda v, n=name: self.apply_parameters({n: v})
                )
            elif param_type == "int":
                w = QSpinBox()
                w.setRange(cfg["min"], cfg["max"])
                if "step" in cfg:
                    w.setSingleStep(cfg["step"])
                elif name == "exposure":
                    w.setSingleStep(100)
                w.setValue(int(self._active_parameters.get(name, cfg.get("default"))))
                w.valueChanged.connect(
                    lambda v, n=name: self.apply_parameters({n: v})
                )
            elif param_type == "float":
                w = QDoubleSpinBox()
                w.setRange(cfg["min"], cfg["max"])
                w.setSingleStep(cfg.get("step", 0.1))
                w.setValue(float(self._active_parameters.get(name, cfg.get("default"))))
                w.valueChanged.connect(
                    lambda v, n=name: self.apply_parameters({n: v})
                )
            elif param_type == "bool":
                w = QCheckBox()
                w.setChecked(bool(self._active_parameters.get(name, cfg.get("default"))))
                w.stateChanged.connect(
                    lambda v, n=name: self.apply_parameters({n: bool(v)})
                )
            else:
                w = QLabel("Unsupported")

            self._basic_controls[name] = w
            basic_layout.addRow(name.title(), w)

        basic_scroll.setWidget(basic_inner)
        tab_widget.addTab(basic_scroll, "Basic")

        # -----------------
        # 2. Advanced Tab
        # -----------------
        adv_widget = QWidget()
        adv_widget.setObjectName("adv_widget")
        adv_layout = QVBoxLayout(adv_widget)
        adv_layout.setContentsMargins(8, 8, 8, 8)
        adv_layout.setSpacing(6)

        # Top Bar: Search, Editable Filter, and Control Buttons
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search features...")
        search_bar.setClearButtonEnabled(True)
        self._search_bar = search_bar
        top_bar.addWidget(search_bar)

        editable_chk = QCheckBox("Editable only")
        editable_chk.setChecked(True)
        self._editable_chk = editable_chk
        top_bar.addWidget(editable_chk)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload parameters from camera")
        refresh_btn.clicked.connect(self.refresh_advanced_parameters)
        top_bar.addWidget(refresh_btn)

        expand_btn = QPushButton("Expand")
        expand_btn.setToolTip("Expand all categories")
        top_bar.addWidget(expand_btn)

        collapse_btn = QPushButton("Collapse")
        collapse_btn.setToolTip("Collapse all categories")
        top_bar.addWidget(collapse_btn)

        adv_layout.addLayout(top_bar)

        # GenICam Tree Widget
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Feature", "Value"])
        tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        tree.setColumnWidth(0, 180)
        tree.setAlternatingRowColors(True)
        adv_layout.addWidget(tree)
        self._advanced_tree = tree

        expand_btn.clicked.connect(tree.expandAll)
        collapse_btn.clicked.connect(tree.collapseAll)
        search_bar.textChanged.connect(self._filter_advanced_tree)
        editable_chk.toggled.connect(self._filter_advanced_tree)

        # Tab change listener to auto-populate if camera becomes available
        tab_widget.currentChanged.connect(self._on_tab_changed)

        tab_widget.addTab(adv_widget, "Advanced")

        # Populate tree if camera is already open
        if self.camera is not None and self.camera.IsOpen():
            self.populate_advanced_tree()
        else:
            self._show_advanced_placeholder("No active camera. Start camera to view and edit features.")

        return tab_widget

    def _on_tab_changed(self, index):
        # Index 1 is Advanced tab
        if index == 1:
            if self.camera is not None and self.camera.IsOpen() and not self._advanced_controls:
                self.populate_advanced_tree()

    def _show_advanced_placeholder(self, text):
        if self._advanced_tree is None:
            return
        if QThread.currentThread() != self._advanced_tree.thread():
            QTimer.singleShot(0, self._advanced_tree, lambda: self._show_advanced_placeholder(text))
            return
        self._advanced_tree.clear()
        self._advanced_controls.clear()
        item = QTreeWidgetItem(self._advanced_tree, [text, ""])
        item.setData(0, Qt.UserRole, "placeholder")
        item.setDisabled(True)

    # ------------------------------------------------
    # DYNAMIC GENICAM POPULATION
    # ------------------------------------------------
    def populate_advanced_tree(self):
        if self._advanced_tree is None:
            return

        # Ensure execution is strictly on the GUI thread owning _advanced_tree
        if QThread.currentThread() != self._advanced_tree.thread():
            QTimer.singleShot(0, self._advanced_tree, self.populate_advanced_tree)
            return

        if self.camera is None or not self.camera.IsOpen():
            self._show_advanced_placeholder("Camera is closed or not initialized.")
            return

        self._advanced_tree.clear()
        self._advanced_controls.clear()

        try:
            nodemap = self.camera.GetNodeMap()
            root = nodemap.GetNode('Root')

            def build_subtree(category_param, parent_item):
                for feat in category_param.Features:
                    node = feat.Node
                    if not genicam.IsImplemented(node) or not genicam.IsAvailable(node):
                        continue

                    itype = node.GetPrincipalInterfaceType()

                    # Sub-category
                    if itype == genicam.intfICategory:
                        cat_item = QTreeWidgetItem(parent_item, [node.Name, ""])
                        cat_item.setData(0, Qt.UserRole, "category")
                        build_subtree(feat, cat_item)
                        if cat_item.childCount() == 0:
                            parent_item.removeChild(cat_item)
                    else:
                        is_writable = genicam.IsWritable(node)
                        is_readable = genicam.IsReadable(node)

                        item = QTreeWidgetItem(parent_item, [node.Name, ""])
                        item.setData(0, Qt.UserRole, "feature")
                        item.setData(0, Qt.UserRole + 1, is_writable)

                        tooltip = ""
                        try:
                            tooltip = node.GetToolTip() or node.GetDescription()
                        except Exception:
                            pass
                        if tooltip:
                            item.setToolTip(0, tooltip)
                            item.setToolTip(1, tooltip)

                        ctrl = None
                        try:
                            # 1. Enumeration
                            if itype == genicam.intfIEnumeration:
                                if is_writable:
                                    cb = QComboBox(self._advanced_tree)
                                    valid_items = []
                                    for sym in getattr(feat, 'Symbolics', []):
                                        try:
                                            entry = feat.GetEntryByName(sym)
                                            if entry and genicam.IsAvailable(entry.Node):
                                                valid_items.append(sym)
                                        except Exception:
                                            valid_items.append(sym)
                                    cb.addItems(valid_items)
                                    try:
                                        cb.setCurrentText(str(feat.Value))
                                    except Exception:
                                        pass
                                    cb.currentTextChanged.connect(
                                        lambda v, p=feat: self._set_advanced_param_value(p, v)
                                    )
                                    ctrl = cb
                                else:
                                    ctrl = QLabel(str(getattr(feat, 'Value', '')), self._advanced_tree)

                            # 2. Integer
                            elif itype == genicam.intfIInteger:
                                val = feat.Value if is_readable else 0
                                if is_writable and (-2147483648 <= val <= 2147483647):
                                    sp = QSpinBox(self._advanced_tree)
                                    sp.setRange(max(-2147483648, feat.Min), min(2147483647, feat.Max))
                                    sp.setValue(val)
                                    try:
                                        inc = feat.Inc
                                        if inc > 0:
                                            sp.setSingleStep(inc)
                                    except Exception:
                                        pass
                                    sp.valueChanged.connect(
                                        lambda v, p=feat: self._set_advanced_param_value(p, int(v))
                                    )
                                    ctrl = sp
                                elif is_writable:
                                    le = QLineEdit(str(val), self._advanced_tree)
                                    le.editingFinished.connect(
                                        lambda p=feat, w=le: self._set_advanced_param_value(p, int(w.text()))
                                    )
                                    ctrl = le
                                else:
                                    ctrl = QLabel(str(val), self._advanced_tree)

                            # 3. Float
                            elif itype == genicam.intfIFloat:
                                if is_writable:
                                    dsp = QDoubleSpinBox(self._advanced_tree)
                                    dsp.setRange(feat.Min, feat.Max)
                                    dsp.setDecimals(2)
                                    dsp.setValue(feat.Value if is_readable else feat.Min)
                                    step_set = False
                                    try:
                                        inc = feat.Inc
                                        if inc > 0:
                                            dsp.setSingleStep(inc)
                                            step_set = True
                                    except Exception:
                                        pass
                                    if not step_set:
                                        if feat.Max > 1000:
                                            dsp.setSingleStep(100.0)
                                        elif feat.Max > 100:
                                            dsp.setSingleStep(1.0)
                                        else:
                                            dsp.setSingleStep(0.1)
                                    dsp.valueChanged.connect(
                                        lambda v, p=feat: self._set_advanced_param_value(p, float(v))
                                    )
                                    ctrl = dsp
                                else:
                                    try:
                                        ctrl = QLabel(f"{feat.Value:.2f}", self._advanced_tree)
                                    except Exception:
                                        ctrl = QLabel(str(getattr(feat, 'Value', '')), self._advanced_tree)

                            # 4. Boolean
                            elif itype == genicam.intfIBoolean:
                                chk = QCheckBox(self._advanced_tree)
                                if is_readable:
                                    chk.setChecked(bool(feat.Value))
                                chk.setEnabled(is_writable)
                                if is_writable:
                                    chk.toggled.connect(
                                        lambda v, p=feat: self._set_advanced_param_value(p, bool(v))
                                    )
                                ctrl = chk

                            # 5. Command
                            elif itype == genicam.intfICommand:
                                btn = QPushButton("Execute", self._advanced_tree)
                                btn.setEnabled(is_writable)
                                if is_writable:
                                    btn.clicked.connect(
                                        lambda _, p=feat: self._execute_advanced_command(p)
                                    )
                                ctrl = btn

                            # 6. String
                            elif itype == genicam.intfIString:
                                if is_writable:
                                    le = QLineEdit(str(getattr(feat, 'Value', '')), self._advanced_tree)
                                    le.editingFinished.connect(
                                        lambda p=feat, w=le: self._set_advanced_param_value(p, w.text())
                                    )
                                    ctrl = le
                                else:
                                    ctrl = QLabel(str(getattr(feat, 'Value', '')), self._advanced_tree)

                        except Exception as e:
                            ctrl = QLabel(f"Err: {e}", self._advanced_tree)

                        if ctrl:
                            self._advanced_tree.setItemWidget(item, 1, ctrl)
                            self._advanced_controls[node.Name] = (ctrl, feat, itype, is_writable)

            for feat in root.Features:
                node = feat.Node
                if not genicam.IsImplemented(node) or not genicam.IsAvailable(node):
                    continue
                if node.GetPrincipalInterfaceType() == genicam.intfICategory:
                    cat_item = QTreeWidgetItem(self._advanced_tree, [node.Name, ""])
                    cat_item.setData(0, Qt.UserRole, "category")
                    build_subtree(feat, cat_item)
                    if cat_item.childCount() == 0:
                        self._advanced_tree.invisibleRootItem().removeChild(cat_item)

            self._filter_advanced_tree()

        except Exception as e:
            print(f"[ERR] Failed populating advanced tree: {e}")
            self._show_advanced_placeholder(f"Error loading camera features: {e}")

    # ------------------------------------------------
    # ADVANCED SEARCH & FILTER
    # ------------------------------------------------
    def _filter_advanced_tree(self):
        if self._advanced_tree is None:
            return

        search_text = self._search_bar.text().strip().lower() if self._search_bar else ""
        editable_only = self._editable_chk.isChecked() if self._editable_chk else False

        def check_item(item):
            role_type = item.data(0, Qt.UserRole)
            if role_type == "placeholder":
                item.setHidden(False)
                return True

            is_category = (role_type == "category")
            is_writable = bool(item.data(0, Qt.UserRole + 1))
            name = item.text(0).lower()

            if not is_category:
                match_search = (not search_text) or (search_text in name)
                match_editable = (not editable_only) or is_writable
                visible = bool(match_search and match_editable)
                item.setHidden(not visible)
                return visible
            else:
                child_visible = False
                for i in range(item.childCount()):
                    child = item.child(i)
                    if check_item(child):
                        child_visible = True

                cat_matches = (search_text in name) and (not editable_only or child_visible)
                is_visible = child_visible or bool(search_text and cat_matches)
                item.setHidden(not is_visible)
                if search_text and child_visible:
                    item.setExpanded(True)
                return is_visible

        for i in range(self._advanced_tree.topLevelItemCount()):
            top_item = self._advanced_tree.topLevelItem(i)
            check_item(top_item)

    # ------------------------------------------------
    # ADVANCED PARAMETER SETTING & SYNC
    # ------------------------------------------------
    def _set_advanced_param_value(self, param, value):
        if not self.camera or not self.camera.IsOpen():
            return

        node_name = param.Node.Name

        try:
            param.Value = value
        except Exception:
            # Some parameters require pausing acquisition
            was_grabbing = False
            try:
                was_grabbing = self.camera.IsGrabbing()
                if was_grabbing:
                    self.camera.StopGrabbing()
                param.Value = value
            except Exception as e:
                print(f"[ERR] Failed setting {node_name} to {value}: {e}")
            finally:
                if was_grabbing:
                    try:
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    except Exception as e:
                        print(f"[ERR] Failed restarting grabbing: {e}")

        # Update converter if PixelFormat was altered
        if node_name == "PixelFormat" and self.converter:
            val_str = str(value)
            if "Mono" in val_str:
                self.converter.OutputPixelFormat = pylon.PixelType_Mono8
            else:
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        # Synchronize corresponding Basic tab control if applicable
        basic_key = GENICAM_TO_BASIC.get(node_name)
        if basic_key:
            self._active_parameters[basic_key] = value
            self._sync_basic_control(basic_key, value)

    def _execute_advanced_command(self, param):
        if not self.camera or not self.camera.IsOpen():
            return
        try:
            param.Execute()
            print(f"[INFO] Executed camera command: {param.Node.Name}")
        except Exception as e:
            print(f"[ERR] Failed executing command {param.Node.Name}: {e}")

    def _sync_basic_control(self, basic_name, val):
        if basic_name not in self._basic_controls:
            return
        ctrl = self._basic_controls[basic_name]
        ctrl.blockSignals(True)
        try:
            if isinstance(ctrl, QComboBox):
                ctrl.setCurrentText(str(val))
            elif isinstance(ctrl, (QSpinBox, QDoubleSpinBox)):
                ctrl.setValue(val)
            elif isinstance(ctrl, QCheckBox):
                ctrl.setChecked(bool(val))
        except Exception:
            pass
        finally:
            ctrl.blockSignals(False)

    def _sync_advanced_control(self, node_name, val):
        if node_name not in self._advanced_controls:
            return
        ctrl, feat, itype, is_writable = self._advanced_controls[node_name]
        if not is_writable:
            return
        ctrl.blockSignals(True)
        try:
            if itype == genicam.intfIEnumeration and isinstance(ctrl, QComboBox):
                ctrl.setCurrentText(str(val))
            elif itype == genicam.intfIInteger and isinstance(ctrl, QSpinBox):
                ctrl.setValue(int(val))
            elif itype == genicam.intfIFloat and isinstance(ctrl, QDoubleSpinBox):
                ctrl.setValue(float(val))
            elif itype == genicam.intfIBoolean and isinstance(ctrl, QCheckBox):
                ctrl.setChecked(bool(val))
            elif itype == genicam.intfIString and isinstance(ctrl, QLineEdit):
                ctrl.setText(str(val))
        except Exception:
            pass
        finally:
            ctrl.blockSignals(False)

    def refresh_advanced_parameters(self):
        if self._advanced_tree is not None and QThread.currentThread() != self._advanced_tree.thread():
            QTimer.singleShot(0, self._advanced_tree, self.refresh_advanced_parameters)
            return

        if not self.camera or not self.camera.IsOpen():
            return

        if not self._advanced_controls:
            self.populate_advanced_tree()
            return

        for node_name, (ctrl, feat, itype, is_writable) in self._advanced_controls.items():
            try:
                if not genicam.IsReadable(feat.Node):
                    continue
                ctrl.blockSignals(True)
                if itype == genicam.intfIEnumeration:
                    if is_writable and isinstance(ctrl, QComboBox):
                        ctrl.setCurrentText(str(feat.Value))
                    elif isinstance(ctrl, QLabel):
                        ctrl.setText(str(feat.Value))
                elif itype == genicam.intfIInteger:
                    if is_writable and isinstance(ctrl, QSpinBox):
                        ctrl.setValue(feat.Value)
                    elif isinstance(ctrl, QLabel):
                        ctrl.setText(str(feat.Value))
                elif itype == genicam.intfIFloat:
                    if is_writable and isinstance(ctrl, QDoubleSpinBox):
                        ctrl.setValue(feat.Value)
                    elif isinstance(ctrl, QLabel):
                        ctrl.setText(f"{feat.Value:.2f}")
                elif itype == genicam.intfIBoolean:
                    if isinstance(ctrl, QCheckBox):
                        ctrl.setChecked(bool(feat.Value))
                elif itype == genicam.intfIString:
                    if is_writable and isinstance(ctrl, QLineEdit):
                        ctrl.setText(str(feat.Value))
                    elif isinstance(ctrl, QLabel):
                        ctrl.setText(str(feat.Value))
                ctrl.blockSignals(False)
            except Exception:
                ctrl.blockSignals(False)

        # Sync Basic tab controls with live camera values
        for basic_name, node_name in BASIC_TO_GENICAM.items():
            if hasattr(self.camera, node_name):
                try:
                    p = getattr(self.camera, node_name)
                    val = p.Value
                    self._active_parameters[basic_name] = val
                    self._sync_basic_control(basic_name, val)
                except Exception:
                    pass

    # ------------------------------------------------
    # BASIC PARAMETER APPLICATION
    # ------------------------------------------------
    def apply_parameters(self, params: dict):
        self._active_parameters.update(params)

        if not self.camera or not self.camera.IsOpen():
            return

        try:
            if "pixel format" in params:
                same_format = self.camera.PixelFormat.Value == params["pixel format"]

                if not same_format:
                    was_grabbing = self.camera.IsGrabbing()
                    if was_grabbing:
                        self.camera.StopGrabbing()

                    self.camera.PixelFormat.Value = params["pixel format"]

                    if "Mono" in params["pixel format"]:
                        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
                    else:
                        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

                    if was_grabbing:
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            if "exposure" in params and hasattr(self.camera, 'ExposureTime'):
                try:
                    self.camera.ExposureTime.Value = params["exposure"]
                except Exception as e:
                    print(f"[ERR] ExposureTime: {e}")

            if "gain" in params and hasattr(self.camera, 'Gain'):
                try:
                    self.camera.Gain.Value = params["gain"]
                except Exception as e:
                    print(f"[ERR] Gain: {e}")

            if "contrast" in params and hasattr(self.camera, 'BslContrast'):
                try:
                    self.camera.BslContrast.Value = params["contrast"]
                except Exception as e:
                    print(f"[ERR] BslContrast: {e}")

            if "hue" in params and hasattr(self.camera, 'BslHue'):
                try:
                    self.camera.BslHue.Value = params["hue"]
                except Exception as e:
                    print(f"[ERR] BslHue: {e}")

            if "saturation" in params and hasattr(self.camera, 'BslSaturation'):
                try:
                    self.camera.BslSaturation.Value = params["saturation"]
                except Exception as e:
                    print(f"[ERR] BslSaturation: {e}")

            if "brightness" in params and hasattr(self.camera, 'BslBrightness'):
                try:
                    self.camera.BslBrightness.Value = params["brightness"]
                except Exception as e:
                    print(f"[ERR] BslBrightness: {e}")

            if "reversex" in params and hasattr(self.camera, 'ReverseX'):
                try:
                    self.camera.ReverseX.Value = params["reversex"]
                except Exception as e:
                    print(f"[ERR] ReverseX: {e}")

            if "reversey" in params and hasattr(self.camera, 'ReverseY'):
                try:
                    self.camera.ReverseY.Value = params["reversey"]
                except Exception as e:
                    print(f"[ERR] ReverseY: {e}")

        except Exception as e:
            print(f"[ERR] Failed to update camera parameters: {e}")

        # Synchronize with Advanced tab
        for basic_key, val in params.items():
            node_name = BASIC_TO_GENICAM.get(basic_key)
            if node_name:
                self._sync_advanced_control(node_name, val)
