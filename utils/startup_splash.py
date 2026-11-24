# ==============================================================================
# VisionView - Startup Loading Splash
# Shows a progress popup while heavy libraries (OpenCV, camera SDKs, PyTorch)
# load, so users know to wait instead of re-launching the app.
# Keep this module lightweight: it must import only PySide6 + stdlib.
# ==============================================================================

import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QProgressBar, QFrame)


class StartupSplash(QWidget):
    """Frameless, always-on-top loading window with a status line and progress bar."""

    def __init__(self, icon_path=None):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFixedSize(440, 170)

        frame = QFrame(self)
        frame.setObjectName("splashFrame")
        frame.setStyleSheet("""
            #splashFrame { background-color: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 8px; }
            QLabel { color: #e0e0e0; background: transparent; }
            #title { font-size: 20px; font-weight: bold; color: #ffffff; }
            #status { font-size: 12px; color: #bdbdbd; }
            #hint { font-size: 11px; color: #8a8a8a; }
            QProgressBar { background-color: #2d2d2d; border: none; border-radius: 4px; height: 8px; }
            QProgressBar::chunk { background-color: #0078d4; border-radius: 4px; }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(12)
        if icon_path and Path(icon_path).exists():
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(str(icon_path)).scaled(
                40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            header.addWidget(icon_label)
            self.setWindowIcon(QIcon(str(icon_path)))
        title = QLabel("VisionView")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        self.status_label = QLabel("Starting...")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)

        hint = QLabel("Loading may take a moment on first launch. Please wait...")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        self._center_on_screen()

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2, geo.center().y() - self.height() // 2)

    def step(self, message, value=None):
        """Update the status text / progress and repaint immediately."""
        self.status_label.setText(message)
        if value is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(value))
        self.repaint()
        QApplication.processEvents()

    def run_in_background(self, message, func, value=None):
        """
        Run a slow, GUI-free call (e.g. importing torch) on a worker thread while
        keeping the splash animated and responsive. Returns func's result and
        re-raises its exception, if any.
        """
        self.status_label.setText(message)
        self.progress.setRange(0, 0)  # indeterminate "busy" animation
        result = {}

        def _target():
            try:
                result["value"] = func()
            except BaseException as e:  # propagate to the caller on the GUI thread
                result["error"] = e

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        while worker.is_alive():
            QApplication.processEvents()
            time.sleep(0.02)

        self.progress.setRange(0, 100)
        if value is not None:
            self.progress.setValue(int(value))
        QApplication.processEvents()

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def finish(self, main_window=None):
        """Close the splash once the main window is visible."""
        self.step("Ready", 100)
        self.close()
        if main_window is not None:
            main_window.raise_()
            main_window.activateWindow()
