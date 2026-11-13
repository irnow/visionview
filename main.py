import sys
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QFrame, QLabel, 
                             QMenu, QMenuBar, QStackedWidget, QSplitter,
                             QToolBar, QLineEdit, QFileDialog, QToolButton,
                             QDockWidget, QMessageBox)
from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtGui import QAction, QIcon, QActionGroup, QKeySequence

from utils.window_module import WindowModule, get_module_icon
from utils.process_canva import ProcessWindow, get_cached_cv_modules
from utils.gui_signal import gui_signal
from utils.directory_viewer import DirectoryViewer
from camera import *

try:
    from qt_material_icons import MaterialIcon
except ImportError:
    MaterialIcon = lambda name: QIcon()

if getattr(sys, 'frozen', False):
    BUNDLE_DIR = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    APP_DIR = Path(sys.executable).resolve().parent
    BASE_DIR = BUNDLE_DIR
    DEFAULT_OUTPUT_DIR = APP_DIR.parent / "media_output"
else:
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_OUTPUT_DIR = BASE_DIR.parent / "media_output"

ICON_DIR = BASE_DIR / "icons"
ARROW_UP_PATH = (ICON_DIR / "arrow_up.png").as_posix()
ARROW_DOWN_PATH = (ICON_DIR / "arrow_down.png").as_posix()

CHECKBOX_STYLESHEET = """
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
QLabel {
    background: transparent;
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
    image: url("__ARROW_UP__");
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
    image: url("__ARROW_DOWN__");
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
    image: url("__ARROW_DOWN__");
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
"""
CHECKBOX_STYLESHEET = CHECKBOX_STYLESHEET.replace("__ARROW_UP__", ARROW_UP_PATH).replace("__ARROW_DOWN__", ARROW_DOWN_PATH)

class VisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionView")
        self.resize(1200, 800)

        # Apply global application style for checkboxes (solid blue fill when checked, no tick sign)
        app_inst = QApplication.instance()
        if app_inst:
            existing_ss = app_inst.styleSheet() or ""
            if "QCheckBox::indicator:checked" not in existing_ss:
                app_inst.setStyleSheet(existing_ss + "\n" + CHECKBOX_STYLESHEET)

        self.setup_menu()
        self.setup_ui()

        self.global_calibration = {'scale': 1.0, 'unit': 'px', 'is_calibrated': False}
        self.active_cameras = [] # active camera name (e.g. 2322315), not active module
        self.dir_popup = None   # save directory popup
        self.module_counter = 0

        gui_signal.closed_module.connect(self.remove_module)
        gui_signal.enlarge_module.connect(self.handle_enlarge)
        gui_signal.module_selected.connect(self.set_active_module)
        gui_signal.update_active_cameras.connect(self.update_active_camera)

        if hasattr(self, 'process_dock'):
            self.process_dock.visibilityChanged.connect(
                lambda visible: self.show_module_process_editor(self.active_module) if visible and self.active_module else None
            )

        # Pre-warm icons and CV module list so adding modules is instantaneous
        try:
            get_module_icon("open_in_full", "white")
            get_module_icon("close", "white")
            get_cached_cv_modules()
        except Exception:
            pass

    def setup_menu(self):
        menubar = self.menuBar()
        self.view_menu = menubar.addMenu("View")
        self.tools_menu = menubar.addMenu("Tools")
        menubar.addMenu("Help")

        self.tools_setup()

        # ---- TOOLBAR ----
        self.toolbar = QToolBar("Camera Tools")
        self.toolbar.setIconSize(QSize(30, 30))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(self.toolbar)

        self.action_add = QAction(MaterialIcon("add_circle"), "Add Module", self)
        self.action_add.triggered.connect(self.create_new_module)

        self.action_start = QAction(MaterialIcon("photo_camera"), "Start Camera", self)
        self.action_start.triggered.connect(lambda: self.trigger_module_action("is_camera"))

        self.action_upload = QAction(MaterialIcon("image"), "Upload Image", self)
        self.action_upload.triggered.connect(lambda: self.trigger_module_action("upload_image"))

        self.action_process = QAction(MaterialIcon("settings"), "Process Editor", self)
        self.action_process.triggered.connect(self.open_process_editor)

        self.action_capture = QAction(MaterialIcon("camera"), "Capture Image", self)
        self.action_capture.setShortcuts([
            QKeySequence("Ctrl+S"),
            QKeySequence("Ctrl+Shift+S"),
        ])
        self.action_capture.setShortcutContext(Qt.ApplicationShortcut)
        self.action_capture.setToolTip("Capture Image (Ctrl+S)")
        self.action_capture.triggered.connect(self.capture_image)
        self.addAction(self.action_capture)

        self.action_record = QAction(MaterialIcon("videocam"), "Record Video", self)
        self.action_record.triggered.connect(self.record_video)

        self.action_stop = QAction(MaterialIcon("stop_circle"), "Stop Video", self)
        self.action_stop.triggered.connect(self.stop_video)

        self.action_fullscreen_module = QAction(MaterialIcon("open_in_full"), "Full Screen Selected Module", self)
        self.action_fullscreen_module.setToolTip("Full Screen Selected Module on Desktop/Window (Esc to return)")
        self.action_fullscreen_module.triggered.connect(self.toggle_selected_module_fullscreen)
        self.action_fullscreen = self.action_fullscreen_module # alias

        self.action_start.setEnabled(False)
        self.action_upload.setEnabled(False)
        self.action_process.setEnabled(False)
        self.action_capture.setEnabled(False)
        self.action_record.setEnabled(False)
        self.action_stop.setEnabled(False)
        self.action_fullscreen_module.setEnabled(False)

        self.toolbar.addAction(self.action_add)
        self.toolbar.addAction(self.action_start)
        self.toolbar.addAction(self.action_upload)
        self.toolbar.addAction(self.action_process)
        self.toolbar.addAction(self.action_capture)
        self.toolbar.addAction(self.action_record)
        self.toolbar.addAction(self.action_stop)
        self.toolbar.addAction(self.action_fullscreen_module)

        self.toolbar.addSeparator()

        # ---- DIRECTORY PATH ----
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.dir_path = str(DEFAULT_OUTPUT_DIR)

        self.dir_edit = QLineEdit(self.dir_path)
        self.dir_edit.setMinimumWidth(250)
        self.dir_edit.editingFinished.connect(self.on_dir_edit_finished)

        # Folder select button
        self.btn_select_dir = QToolButton()
        self.btn_select_dir.setIcon(MaterialIcon("folder_open"))
        self.btn_select_dir.setToolTip("Select Output Directory")
        self.btn_select_dir.clicked.connect(self.select_directory)

        self.toolbar.addWidget(self.btn_select_dir)
        self.toolbar.addWidget(self.dir_edit)

        # ---- MEASUREMENT & METROLOGY TOOLBAR ----
        self.measure_toolbar = QToolBar("Inspection & Metrology Tools")
        self.measure_toolbar.setIconSize(QSize(26, 26))
        self.measure_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(self.measure_toolbar)

        self.measure_toolbar.addAction(self.zoom_tool)
        self.measure_toolbar.addAction(self.length_tool)
        self.measure_toolbar.addAction(self.caliper_tool)
        self.measure_toolbar.addAction(self.circle_tool)
        self.measure_toolbar.addAction(self.circle_3p_tool)
        self.measure_toolbar.addAction(self.angle_tool)
        self.measure_toolbar.addAction(self.square_tool)
        self.measure_toolbar.addAction(self.polygon_tool)
        self.measure_toolbar.addSeparator()
        self.measure_toolbar.addAction(self.calibrate_tool)
        self.measure_toolbar.addSeparator()
        self.measure_toolbar.addAction(self.clear_measure_tool)

        self.active_module = None

    def setup_ui(self):
        # 1. Setup the Central Widget (The Module Canvas)
        # In QMainWindow, the Central Widget is anchor for docks
        self.central_container = QFrame()
        self.setCentralWidget(self.central_container)
        
        # Re-link the canvas layout to the central container
        self.canvas_layout = QGridLayout(self.central_container)
        
        # Pre-configure the grid to have 3 rows and 2 columns
        for i in range(3):
            self.canvas_layout.setRowStretch(i, 1)
        for j in range(2):
            self.canvas_layout.setColumnStretch(j, 1)

        self.central_container.setContextMenuPolicy(Qt.CustomContextMenu)
        self.central_container.customContextMenuRequested.connect(self.show_canvas_menu)
        
        self.modules = []

        # Create the StackedWidget to live inside the dock
        self.right_panel = QStackedWidget()
        self.right_panel.setMinimumWidth(300)
        self.right_panel.setStyleSheet("background-color: palette(window);")

        # Placeholder for right-side settings
        self.default_setting = QLabel("No active camera\n(Click active camera to adjust)")
        self.default_setting.setAlignment(Qt.AlignCenter)
        self.right_panel.addWidget(self.default_setting)

        self.view_setup()

    def view_setup(self):
        # 1. Setup the Dynamic Right Panel (Camera Parameters)
        self.settings_dock = QDockWidget("Camera Parameters", self)
        self.settings_dock.setObjectName("SettingsDock") # for saving layout state later
        self.settings_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.settings_dock.setMinimumWidth(320)
        self.settings_dock.setWidget(self.right_panel)

        # 2. Save Directory Dock
        self.dir_dock = QDockWidget("Saved Directory", self)
        self.dir_dock.setObjectName("DirDock")
        self.dir_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.dir_viewer = DirectoryViewer(self.dir_edit.text(), self)
        self.dir_dock.setWidget(self.dir_viewer)

        # 3. Process Editor Dock (Attached at bottom by default)
        self.process_dock = QDockWidget("Process Editor", self)
        self.process_dock.setObjectName("ProcessDock")
        self.process_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        
        self.process_panel = QStackedWidget()
        self.default_process_label = QLabel("No active module\n(Select or add a module to edit process pipeline)")
        self.default_process_label.setAlignment(Qt.AlignCenter)
        self.default_process_label.setStyleSheet("color: #888; font-size: 13px;")
        self.process_panel.addWidget(self.default_process_label)
        self.process_dock.setWidget(self.process_panel)

        # Add Docks: Camera Parameters & Saved Directory on Right by default, Process Editor on Bottom by default
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dir_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.process_dock)

        self.resizeDocks([self.process_dock], [260], Qt.Vertical)

        self.dir_dock.hide()
        self.process_dock.hide()

        self.view_menu.addAction(self.process_dock.toggleViewAction())
        self.view_menu.addAction(self.dir_dock.toggleViewAction())
        self.view_menu.addAction(self.settings_dock.toggleViewAction())

    def tools_setup(self):
        self.tool_group = QActionGroup(self)

        self.zoom_tool = QAction(MaterialIcon("search"), "Zoom", self)
        self.zoom_tool.setData("zoom")
        self.zoom_tool.setCheckable(True)
        self.zoom_tool.setChecked(True)
        self.zoom_tool.setToolTip("Zoom Tool (Drag ROI to zoom in, Right-click to reset)")
        self.tool_group.addAction(self.zoom_tool)
        self.tools_menu.addAction(self.zoom_tool)

        self.length_tool = QAction(MaterialIcon("straighten"), "Length", self)
        self.length_tool.setData("length")
        self.length_tool.setCheckable(True)
        self.length_tool.setToolTip("Length Tool (Drag between points to measure linear distance)")
        self.tool_group.addAction(self.length_tool)
        self.tools_menu.addAction(self.length_tool)

        self.caliper_tool = QAction(MaterialIcon("linear_scale"), "Edge Caliper (Sub-Pixel)", self)
        self.caliper_tool.setData("caliper")
        self.caliper_tool.setCheckable(True)
        self.caliper_tool.setToolTip("Sub-Pixel Edge Caliper (Drag across edge to detect peak gradients and measure features)")
        self.tool_group.addAction(self.caliper_tool)
        self.tools_menu.addAction(self.caliper_tool)

        # Circle tools menu & actions
        self.circle_menu = self.tools_menu.addMenu("Circle")
        self.circle_tool = QAction(MaterialIcon("radio_button_unchecked"), "Circle (Center-Radius)", self)
        self.circle_tool.setData("circle")
        self.circle_tool.setCheckable(True)
        self.circle_tool.setToolTip("Circle Tool (Drag from center outward to radius)")
        self.tool_group.addAction(self.circle_tool)
        self.circle_menu.addAction(self.circle_tool)

        self.circle_3p_tool = QAction(MaterialIcon("circle"), "Circle (3-Point Fit)", self)
        self.circle_3p_tool.setData("circle_3p")
        self.circle_3p_tool.setCheckable(True)
        self.circle_3p_tool.setToolTip("Circle (3-Point) (Click 3 points on rim/hole to fit circumcircle)")
        self.tool_group.addAction(self.circle_3p_tool)
        self.circle_menu.addAction(self.circle_3p_tool)

        # Angle tool
        self.angle_tool = QAction(MaterialIcon("square_foot"), "Angle / Protractor", self)
        self.angle_tool.setData("angle")
        self.angle_tool.setCheckable(True)
        self.angle_tool.setToolTip("Angle Protractor (Click Point 1 -> Vertex -> Point 2 to measure interior angle)")
        self.tool_group.addAction(self.angle_tool)
        self.tools_menu.addAction(self.angle_tool)

        # Area tools menu & actions
        self.area_menu = self.tools_menu.addMenu("Area")
        self.square_tool = QAction(MaterialIcon("crop_square"), "Square", self)
        self.square_tool.setData("square")
        self.square_tool.setCheckable(True)
        self.square_tool.setToolTip("Square / Rectangle Area (Drag bounding box)")
        self.tool_group.addAction(self.square_tool)
        self.area_menu.addAction(self.square_tool)

        self.polygon_tool = QAction(MaterialIcon("polyline"), "Polygon", self)
        self.polygon_tool.setData("polygon")
        self.polygon_tool.setCheckable(True)
        self.polygon_tool.setToolTip("Polygon Area (Click vertices, double-click or click start point to close)")
        self.tool_group.addAction(self.polygon_tool)
        self.area_menu.addAction(self.polygon_tool)

        self.tools_menu.addSeparator()

        # Calibration
        self.calibrate_tool = QAction(MaterialIcon("tune"), "Calibrate Scale (px -> mm)", self)
        self.calibrate_tool.setData("calibrate")
        self.calibrate_tool.setCheckable(True)
        self.calibrate_tool.setToolTip("Scale Calibration (Drag line across known dimension e.g. 10 mm ruler)")
        self.tool_group.addAction(self.calibrate_tool)
        self.tools_menu.addAction(self.calibrate_tool)
        self.tools_menu.addSeparator()

        # Clear measurements
        self.clear_measure_tool = QAction(MaterialIcon("delete"), "Clear Measurements", self)
        self.clear_measure_tool.setToolTip("Clear all measurements on active module")
        self.clear_measure_tool.triggered.connect(self.clear_active_measurements)
        self.tools_menu.addAction(self.clear_measure_tool)

        self.tool_group.triggered.connect(self.update_tool_mode)

    def update_tool_mode(self, action):
        mode = action.data()
        if not mode:
            raw_mode = action.text().lower() 
            if "template" in raw_mode:
                mode = "template"
            elif "caliper" in raw_mode:
                mode = "caliper"
            elif "3-point" in raw_mode or "circle_3p" in raw_mode:
                mode = "circle_3p"
            elif "circle" in raw_mode:
                mode = "circle"
            elif "angle" in raw_mode:
                mode = "angle"
            elif "calibrate" in raw_mode:
                mode = "calibrate"
            elif "polygon" in raw_mode:
                mode = "polygon"
            elif "square" in raw_mode or "rectangle" in raw_mode:
                mode = "square"
            elif "length" in raw_mode:
                mode = "length"
            else:
                mode = "zoom"

        for mod in self.modules:
            mod.current_tool = mode
            if hasattr(mod.label, 'cancel_all_in_progress'):
                mod.label.cancel_all_in_progress()
            elif hasattr(mod.label, 'cancel_polygon'):
                mod.label.cancel_polygon()

    def clear_active_measurements(self):
        if hasattr(self, 'active_module') and self.active_module:
            self.active_module.measurements = []
            if hasattr(self.active_module.label, 'cancel_all_in_progress'):
                self.active_module.label.cancel_all_in_progress()
            self.active_module.label.update()

    def open_save_directory_popup(self):
        """Now acts as a 'Show/Focus' trigger for the dock"""
        if self.dir_dock.isHidden():
            self.dir_dock.show()
        self.dir_dock.raise_()
        self.dir_dock.activateWindow()

    def show_canvas_menu(self, pos):
        if len(self.modules) >= 6:
            return # Max 6 modules limit

        context_menu = QMenu(self)
        add_module = context_menu.addAction("Add Module")
        
        action = context_menu.exec(self.central_container.mapToGlobal(pos))
        if action == add_module:
            self.create_new_module()

    def create_new_module(self):
        count = len(self.modules)
        if count >= 6:
            return

        # 1. Restore any desktop fullscreen window
        for mod in self.modules:
            if getattr(mod, 'is_fullscreen_window', False):
                mod.exit_fullscreen_window()

        new_index = len(self.modules) + 1
        new_mod = WindowModule(self, new_index)
        
        checked = self.tool_group.checkedAction() if hasattr(self, 'tool_group') else None
        if checked:
            new_mod.current_tool = checked.data() or "zoom"
        else:
            new_mod.current_tool = "zoom"

        if hasattr(self, 'global_calibration') and self.global_calibration.get('is_calibrated'):
            new_mod.apply_calibration(self.global_calibration.copy())

        self.modules.append(new_mod)
        # Restore any existing enlarged module to smaller module and rearrange into normal grid
        self.update_grid_layout()
        self.set_active_module(new_mod)

    def set_active_module(self, module):
        self.active_module = module
        self.action_start.setEnabled(True)
        self.action_upload.setEnabled(True)
        if hasattr(self, 'action_fullscreen_module'):
            self.action_fullscreen_module.setEnabled(True)
            if getattr(module, 'is_fullscreen_window', False):
                self.action_fullscreen_module.setIcon(MaterialIcon("fullscreen_exit"))
                self.action_fullscreen_module.setText("Exit Full Screen")
            else:
                self.action_fullscreen_module.setIcon(MaterialIcon("open_in_full"))
                self.action_fullscreen_module.setText("Full Screen Selected Module")

        for mod in self.modules:
            mod.setStyleSheet("""
            QFrame#moduleFrame {
                background-color:#2b2b2b;
                border:1px solid #555;
            }
            """)

        module.setStyleSheet("""
        QFrame#moduleFrame {
            background-color:#2b2b2b;
            border:2px solid #00aaff;
        }
        """)

        if module:
            self.action_process.setEnabled(True)
            if hasattr(self, 'process_panel'):
                self.show_module_process_editor(module, force_show=False)

        if hasattr(module, "worker") and module.worker:
            self.action_capture.setEnabled(True)

            if getattr(module, "is_recording", False):
                self.action_record.setEnabled(False)
                self.action_stop.setEnabled(True)
            else:
                self.action_record.setEnabled(True)
                self.action_stop.setEnabled(False)

            if not module.static_image:
                if not hasattr(module, "camera_panel") or module.camera_panel is None:
                    module.camera_panel = module.worker.camera_object.parameter_widget()
                    self.right_panel.addWidget(module.camera_panel)
                self.right_panel.setCurrentWidget(module.camera_panel)
            else:
                self.right_panel.setCurrentWidget(self.default_setting)
        else:
            self.right_panel.setCurrentWidget(self.default_setting)
            self.action_capture.setEnabled(False)
            self.action_record.setEnabled(False)
            self.action_stop.setEnabled(False)
            if not module:
                self.action_process.setEnabled(False)
                if hasattr(self, 'process_panel'):
                    self.process_panel.setCurrentWidget(self.default_process_label)

    def open_process_editor(self):
        if not self.active_module and self.modules:
            self.set_active_module(self.modules[0])

        if hasattr(self, 'process_dock'):
            if self.process_dock.isHidden():
                self.process_dock.show()
            self.process_dock.raise_()
            self.process_dock.activateWindow()
        if self.active_module:
            self.show_module_process_editor(self.active_module, force_show=True)

    def show_module_process_editor(self, module, force_show=False):
        if not module:
            return
        if force_show and hasattr(self, 'process_dock'):
            if self.process_dock.isHidden():
                self.process_dock.show()
            self.process_dock.raise_()
            self.process_dock.activateWindow()

        if hasattr(self, 'process_panel'):
            if getattr(module, 'process_window', None) is None:
                module.process_window = ProcessWindow(module)
                if getattr(module, 'saved_nodes', None):
                    module.process_window.canvas.restore_setup(module.saved_nodes)
                else:
                    module.process_window.send_to_worker()
            if self.process_panel.indexOf(module.process_window) == -1:
                self.process_panel.addWidget(module.process_window)
            self.process_panel.setCurrentWidget(module.process_window)

    def trigger_module_action(self, option: str):
        if not self.active_module and self.modules:
            self.set_active_module(self.modules[0])

        if self.active_module:
            # getattr find the function inside the module using the string name
            action_method = getattr(self.active_module, option, None)
            
            if callable(action_method):
                action_method()
            else:
                print(f"Error: Module does not have a method named '{option}'")
        else:
            print(f"Action '{option}' failed: Please select or create a module first.")

    def capture_image(self):
        if self.active_module:
            self.open_save_directory_popup()
            save_dir = self.dir_edit.text()
            self.active_module.capture_image(save_dir)

    def record_video(self):
        if self.active_module:
            self.open_save_directory_popup()
            save_dir = self.dir_edit.text()
            self.active_module.record_video(save_dir)

            self.action_stop.setEnabled(True)
            self.action_record.setEnabled(False)

    def stop_video(self):
        if self.active_module:
            self.active_module.stop_video()

            self.action_stop.setEnabled(False)
            self.action_record.setEnabled(True)

    def set_output_directory(self, folder):
        if not folder:
            return
        self.dir_path = folder
        if self.dir_edit.text() != folder:
            self.dir_edit.setText(folder)

        if hasattr(self, 'dir_viewer') and self.dir_viewer is not None:
            self.dir_viewer.directory = folder
            if hasattr(self.dir_viewer, 'watcher'):
                old_paths = self.dir_viewer.watcher.directories()
                if old_paths:
                    self.dir_viewer.watcher.removePaths(old_paths)
                if Path(folder).exists():
                    self.dir_viewer.watcher.addPath(folder)
            self.dir_viewer.update_file_list()

    def on_dir_edit_finished(self):
        folder = self.dir_edit.text().strip()
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            self.set_output_directory(folder)

    def select_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.dir_edit.text())
        if folder:
            self.set_output_directory(folder)

    def remove_module(self, module_to_remove):
        if getattr(module_to_remove, 'is_fullscreen_window', False):
            module_to_remove.exit_fullscreen_window()

        # Remove from active camera list if needed
        if hasattr(module_to_remove, "serial_info") and module_to_remove.serial_info in self.active_cameras:
            self.active_cameras.remove(module_to_remove.serial_info)

        # Remove parameter panel from right_panel if present
        if hasattr(module_to_remove, "camera_panel") and module_to_remove.camera_panel:
            self.right_panel.removeWidget(module_to_remove.camera_panel)
            module_to_remove.camera_panel.deleteLater()
            module_to_remove.camera_panel = None

        # Remove process_window from process_panel if present
        if hasattr(module_to_remove, "process_window") and module_to_remove.process_window:
            if hasattr(self, 'process_panel'):
                self.process_panel.removeWidget(module_to_remove.process_window)

        # Remove from module list
        if module_to_remove in self.modules:
            self.modules.remove(module_to_remove)

        # Remove from layout
        self.canvas_layout.removeWidget(module_to_remove)

        # Delete widget
        module_to_remove.deleteLater()

        # Re-index all remaining modules so numbering is always continuous: 1, 2, ..., N
        for i, mod in enumerate(self.modules, start=1):
            if hasattr(mod, 'update_index'):
                mod.update_index(i)
            else:
                mod.index = i
                if hasattr(mod, 'title_label'):
                    mod.title_label.setText(f"Module {i}")

        # If the active module was removed
        if module_to_remove == self.active_module:
            self.active_module = None
            if self.modules:
                self.set_active_module(self.modules[-1])
            else:
                self.right_panel.setCurrentWidget(self.default_setting)
                self.action_start.setEnabled(False)
                self.action_upload.setEnabled(False)
                self.action_process.setEnabled(False)
                self.action_capture.setEnabled(False)
                self.action_record.setEnabled(False)
                self.action_stop.setEnabled(False)

        # If ALL modules deleted
        if len(self.modules) == 0:
            self.active_module = None
            self.right_panel.setCurrentWidget(self.default_setting)
            if hasattr(self, 'process_panel'):
                self.process_panel.setCurrentWidget(self.default_process_label)
            self.action_start.setEnabled(False)
            self.action_upload.setEnabled(False)
            self.action_process.setEnabled(False)
            self.action_capture.setEnabled(False)
            self.action_record.setEnabled(False)
            self.action_stop.setEnabled(False)

            if hasattr(self, 'action_fullscreen_module'):
                self.action_fullscreen_module.setEnabled(False)

        self.update_grid_layout()

    def update_grid_layout(self):
        """Refreshes the position of all modules in the canvas grid and resets enlargement."""
        grid_idx = 0
        for mod in self.modules:
            if getattr(mod, 'is_fullscreen_window', False):
                continue
            mod.is_enlarged = False
            if hasattr(mod, 'btn_enlarge'):
                mod.btn_enlarge.setIcon(get_module_icon("open_in_full", "white"))
                mod.btn_enlarge.setToolTip("Enlarge in Panel Canvas (Esc to return)")
            row = grid_idx // 2
            col = grid_idx % 2
            self.canvas_layout.addWidget(mod, row, col, 1, 1)
            mod.show()
            grid_idx += 1

    def update_active_camera(self, camera_info):
        self.active_cameras.append(camera_info)

    def toggle_selected_module_fullscreen(self):
        """Fullscreens the selected module on full desktop / window"""
        if not self.active_module and self.modules:
            self.set_active_module(self.modules[0])

        if self.active_module:
            if hasattr(self.active_module, 'toggle_fullscreen_window'):
                self.active_module.toggle_fullscreen_window()

    def handle_enlarge(self, module, enlarge):
        if module not in self.modules:
            return
        if enlarge:
            for m in self.modules:
                if m != module:
                    m.is_enlarged = False
                    if hasattr(m, 'btn_enlarge'):
                        m.btn_enlarge.setIcon(get_module_icon("open_in_full", "white"))
                        m.btn_enlarge.setToolTip("Enlarge in Panel Canvas (Esc to return)")
            self.enlarge_module(module)
        else:
            self.update_grid_layout()

    def enlarge_module(self, module_to_enlarge):
        # Hide all other modules in canvas and reset their enlarge state
        for mod in self.modules:
            if mod != module_to_enlarge:
                mod.is_enlarged = False
                if hasattr(mod, 'btn_enlarge'):
                    mod.btn_enlarge.setIcon(get_module_icon("open_in_full", "white"))
                    mod.btn_enlarge.setToolTip("Enlarge in Panel Canvas (Esc to return)")
                mod.hide()
        
        # Maximize target module to top in fullscreen of placeholder (spanning all 3 rows and 2 cols)
        module_to_enlarge.is_enlarged = True
        if hasattr(module_to_enlarge, 'btn_enlarge'):
            module_to_enlarge.btn_enlarge.setIcon(get_module_icon("fullscreen_exit", "white"))
            module_to_enlarge.btn_enlarge.setToolTip("Restore Grid (Esc)")
        module_to_enlarge.show()
        module_to_enlarge.raise_()
        self.canvas_layout.addWidget(module_to_enlarge, 0, 0, 3, 2)
        self.set_active_module(module_to_enlarge)

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_S or event.text().lower() == 's') and (event.modifiers() & Qt.ControlModifier):
            if self.active_module and getattr(self.active_module, 'current_frame', None) is not None:
                self.capture_image()
            elif hasattr(self, 'action_capture') and self.action_capture.isEnabled():
                self.capture_image()
            event.accept()
            return

        if event.key() == Qt.Key_Escape:
            # 1. If any module is in fullscreen window on desktop, restore it to canvas grid
            fs_mod = next((m for m in self.modules if getattr(m, 'is_fullscreen_window', False)), None)
            if fs_mod:
                fs_mod.exit_fullscreen_window()
                event.accept()
                return

            # 2. If any module is enlarged in the panel canvas, restore normal canvas grid
            enlarged = [m for m in self.modules if getattr(m, 'is_enlarged', False)]
            if enlarged:
                self.update_grid_layout()
                event.accept()
                return

        if event.key() == Qt.Key_F11:
            self.toggle_maximized()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        print("Shutting down modules...")

        for mod in self.modules:
            if getattr(mod, 'is_fullscreen_window', False):
                mod.close()
            mod.close_event_logic()

        QApplication.processEvents()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(app.styleSheet() + "\n" + CHECKBOX_STYLESHEET)
    window = VisionApp()
    window.showMaximized()
    sys.exit(app.exec())