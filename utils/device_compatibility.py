# ==============================================================================
# VisionView - Device Hardware & Library Compatibility Checker
# Manages hardware detection, PyTorch & CUDA environment checks, and installation
# ==============================================================================

import os
import sys
import shutil
import subprocess
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtWidgets import (
    QApplication, QDialog, QMessageBox, QCheckBox, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QPlainTextEdit
)

_COMPATIBILITY_CACHE = None


def detect_gpu_hardware():
    """
    Detects all video controllers / GPUs on the machine, identifying
    specifically whether an NVIDIA GPU is available for CUDA deep learning acceleration.
    Returns:
        dict with keys: 'has_gpu', 'has_nvidia', 'nvidia_gpus', 'all_gpus', 'driver_version'
    """
    if os.environ.get("VISIONVIEW_FORCE_NO_GPU", "0") == "1":
        return {
            'has_gpu': False,
            'has_nvidia': False,
            'nvidia_gpus': [],
            'all_gpus': [],
            'driver_version': None
        }

    forced_gpu = os.environ.get("VISIONVIEW_FORCE_GPU_NAME")
    if forced_gpu:
        is_nv = any(t in forced_gpu.lower() for t in ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla'])
        return {
            'has_gpu': True,
            'has_nvidia': is_nv,
            'nvidia_gpus': [forced_gpu] if is_nv else [],
            'all_gpus': [forced_gpu],
            'driver_version': "550.00"
        }

    info = {
        'has_gpu': False,
        'has_nvidia': False,
        'nvidia_gpus': [],
        'all_gpus': [],
        'driver_version': None
    }

    # 1. Native Windows Registry scan (instantaneous, 0ms overhead)
    if sys.platform == 'win32':
        try:
            import winreg
            key_path = r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as k:
                num_subkeys = winreg.QueryInfoKey(k)[0]
                for i in range(num_subkeys):
                    sub = winreg.EnumKey(k, i)
                    try:
                        with winreg.OpenKey(k, sub) as sk:
                            desc, _ = winreg.QueryValueEx(sk, 'DriverDesc')
                            if desc and desc not in info['all_gpus']:
                                info['all_gpus'].append(desc)
                    except Exception:
                        pass
        except Exception:
            pass

    # 2. Query nvidia-smi if available (provides accurate driver version and active GPU state)
    smi = shutil.which('nvidia-smi') or (r'C:\Windows\system32\nvidia-smi.exe' if sys.platform == 'win32' else None)
    if smi and os.path.exists(smi):
        try:
            creationflags = 0x08000000 if sys.platform == 'win32' else 0
            out = subprocess.check_output(
                [smi, '--query-gpu=name,driver_version', '--format=csv,noheader'],
                text=True, timeout=2,
                creationflags=creationflags
            )
            lines = [line.strip() for line in out.strip().split('\n') if line.strip()]
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                name = parts[0]
                if name and name not in info['nvidia_gpus']:
                    info['nvidia_gpus'].append(name)
                if len(parts) > 1 and not info['driver_version']:
                    info['driver_version'] = parts[1]
        except Exception:
            pass

    # 3. Match NVIDIA keywords across all detected display adapters
    nvidia_keywords = ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla', 'titan', 'a100', 'h100', 't4', 'l4']
    for gpu in info['all_gpus']:
        lower = gpu.lower()
        if any(term in lower for term in nvidia_keywords):
            if gpu not in info['nvidia_gpus']:
                info['nvidia_gpus'].append(gpu)

    info['has_nvidia'] = len(info['nvidia_gpus']) > 0
    info['has_gpu'] = len(info['all_gpus']) > 0 or info['has_nvidia']
    return info


def get_device_setting(key, default=False):
    """Retrieves a persisted setting from QSettings."""
    try:
        s = QSettings("VisionView", "VisionView")
        val = s.value(key, default)
        if isinstance(default, bool):
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)
        return val
    except Exception:
        return default


def set_device_setting(key, value):
    """Persists a setting into QSettings."""
    try:
        s = QSettings("VisionView", "VisionView")
        s.setValue(key, value)
    except Exception:
        pass


def _log_import_failure(name):
    """Record the full traceback of a failed AI import (the frozen app has no console)."""
    import traceback
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    try:
        from utils.ai_runtime import write_log
        write_log(f"Failed to import {name}:\n{tb}")
    except Exception:
        pass


def check_deep_learning_support(force_refresh=False):
    """
    Checks whether PyTorch, Torchvision, and AI inference can run on this device.
    Returns:
        tuple (is_supported: bool, message: str, details: dict)
    """
    global _COMPATIBILITY_CACHE
    if _COMPATIBILITY_CACHE is not None and not force_refresh:
        return _COMPATIBILITY_CACHE

    # Allow testing unsupported state via environment variable
    if os.environ.get("VISIONVIEW_FORCE_NO_TORCH", "0") == "1":
        res = (
            False,
            "Deep Learning is disabled by environment override (VISIONVIEW_FORCE_NO_TORCH=1).",
            {"torch": False, "torchvision": False, "cuda": False, "device": "None"}
        )
        _COMPATIBILITY_CACHE = res
        return res

    details = {
        "torch": False,
        "torchvision": False,
        "cuda": False,
        "device": "CPU",
        "reason": ""
    }

    # 1. Check PyTorch installation
    try:
        import torch
        details["torch"] = True
        details["torch_version"] = torch.__version__
    except ImportError as e:
        _log_import_failure("torch")
        res = (
            False,
            "The PyTorch ('torch') deep learning library is not installed or bundled in this application build."
            + (f" ({e})" if getattr(e, "name", None) != "torch" else ""),
            details
        )
        _COMPATIBILITY_CACHE = res
        return res
    except Exception as e:
        _log_import_failure("torch")
        res = (
            False,
            f"PyTorch failed to load on this device: {e}",
            details
        )
        _COMPATIBILITY_CACHE = res
        return res

    # 2. Check Torchvision installation
    try:
        import torchvision
        details["torchvision"] = True
        details["torchvision_version"] = torchvision.__version__
    except ImportError:
        _log_import_failure("torchvision")
        res = (
            False,
            "The 'torchvision' computer vision library is not installed on this system.",
            details
        )
        _COMPATIBILITY_CACHE = res
        return res
    except Exception as e:
        _log_import_failure("torchvision")
        res = (
            False,
            f"Torchvision failed to load on this device: {e}",
            details
        )
        _COMPATIBILITY_CACHE = res
        return res

    # 3. Check CPU instruction execution (catches AVX/AVX2 instruction crashes on older CPUs)
    try:
        t = torch.zeros(1) + 1.0
        if float(t.item()) != 1.0:
            raise RuntimeError("CPU tensor computation returned unexpected result.")
    except Exception as e:
        res = (
            False,
            f"Your CPU cannot execute PyTorch operations (missing required AVX instruction set): {e}",
            details
        )
        _COMPATIBILITY_CACHE = res
        return res

    # 4. Check CUDA / GPU availability
    try:
        if torch.cuda.is_available():
            details["cuda"] = True
            details["device"] = torch.cuda.get_device_name(0)
        else:
            details["cuda"] = False
            details["device"] = "CPU (No NVIDIA CUDA GPU active in PyTorch)"
    except Exception:
        details["cuda"] = False
        details["device"] = "CPU"

    # 5. Check RF-DETR framework availability
    try:
        import rfdetr
        details["rfdetr"] = True
    except Exception:
        _log_import_failure("rfdetr")
        details["rfdetr"] = False

    res = (True, "Supported", details)
    try:
        from utils.ai_runtime import write_log
        write_log(f"Deep learning check passed: {details}")
    except Exception:
        pass
    _COMPATIBILITY_CACHE = res
    return res


class TorchInstallWorker(QThread):
    """
    Installs PyTorch + Torchvision for this device.
    - Frozen VisionView.exe: builds a side-by-side runtime (ai_runtime/) with uv,
      because the exe has no Python interpreter or pip of its own.
    - Running from source: installs into the current Python interpreter.
    The CUDA wheel index is chosen from the installed NVIDIA driver version.
    """
    progress_output = Signal(str)
    status_changed = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, python_exe=None, parent=None, driver_version=None):
        super().__init__(parent)
        self.python_exe = python_exe or sys.executable
        self.driver_version = driver_version
        self._is_cancelled = False
        self._process = None

    def cancel(self):
        self._is_cancelled = True
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass

    def _run_command(self, cmd, env=None):
        """Run one command, streaming output to the log. Returns the exit code."""
        self.progress_output.emit(f"\n> {subprocess.list2cmdline([str(c) for c in cmd])}")
        creationflags = 0x08000000 if sys.platform == "win32" else 0
        # The frozen app's bootloader calls SetDllDirectory(_internal), which child
        # processes inherit: torch in the runtime would then load the app's bundled
        # MSVC DLLs instead of its own/system ones. Reset it while spawning.
        reset_dll_dir = sys.platform == "win32" and getattr(sys, "frozen", False)
        if reset_dll_dir:
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        try:
            self._process = subprocess.Popen(
                [str(c) for c in cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags
            )
        finally:
            if reset_dll_dir:
                ctypes.windll.kernel32.SetDllDirectoryW(getattr(sys, "_MEIPASS", None))
        for line in iter(self._process.stdout.readline, ''):
            if self._is_cancelled:
                break
            stripped = line.strip()
            if stripped:
                self.progress_output.emit(stripped)
        self._process.stdout.close()
        return self._process.wait()

    def _build_plan(self):
        """Returns (steps, env, install_root or None)."""
        from utils import ai_runtime

        driver = self.driver_version
        if driver is None:
            driver = detect_gpu_hardware().get("driver_version")
        index_tag = ai_runtime.select_torch_index(driver)
        numpy_version = ai_runtime.bundled_numpy_version()
        self.progress_output.emit(
            f"NVIDIA driver: {driver or 'not detected'} -> PyTorch wheel index: {index_tag}")
        self.progress_output.emit(f"NumPy pinned to: {numpy_version or 'unpinned'}")

        if not ai_runtime.should_use_runtime():
            return ai_runtime.build_source_plan(self.python_exe, index_tag, numpy_version), None, None

        root = ai_runtime.get_install_root()
        self.progress_output.emit(f"AI runtime folder: {root}")
        free = ai_runtime.free_space_gb(root)
        if free is not None and free < ai_runtime.MIN_FREE_GB:
            raise RuntimeError(
                f"Not enough disk space: {free:.1f} GB free, about {ai_runtime.MIN_FREE_GB} GB is needed.")

        uv = ai_runtime.find_uv()
        if uv is None:
            self.status_changed.emit("Downloading package installer (uv)...")
            uv = ai_runtime.download_uv(root / "bin", log=self.progress_output.emit)
        self.progress_output.emit(f"Using uv: {uv}")

        steps = ai_runtime.build_runtime_plan(uv, root, index_tag, numpy_version)
        return steps, ai_runtime.uv_environment(root), root

    def run(self):
        self.status_changed.emit("Preparing installation...")
        try:
            steps, env, root = self._build_plan()
            warnings = []
            for step in steps:
                if self._is_cancelled:
                    break
                self.status_changed.emit(step.status)
                ret_code = self._run_command(step.resolve(), env=env)
                if self._is_cancelled:
                    break
                if ret_code != 0:
                    if step.optional:
                        warnings.append(f"{step.status.rstrip('.')} failed (exit code {ret_code}).")
                        self.progress_output.emit(f"[warning] optional step failed, continuing.")
                        continue
                    self.finished.emit(False, f"{step.status.rstrip('.')} failed (exit code {ret_code}).")
                    return

            if self._is_cancelled:
                self.finished.emit(False, "Installation was cancelled by user.")
                return

            if root is not None:
                # Free the download cache (several GB of wheels) once installed.
                shutil.rmtree(root / "cache", ignore_errors=True)

            message = "PyTorch and Torchvision installed successfully!"
            if warnings:
                message += " (" + " ".join(warnings) + ")"
            self.finished.emit(True, message)
        except Exception as e:
            self.finished.emit(False, f"Installation failed: {e}")


class TorchInstallDialog(QDialog):
    def __init__(self, parent=None, gpu_name="NVIDIA GPU", driver_version=None):
        super().__init__(parent)
        self.setWindowTitle("PyTorch & CUDA Setup")
        self.resize(580, 420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header Title
        title_label = QLabel("<b>Installing PyTorch & Torchvision</b>")
        title_label.setStyleSheet("font-size: 15px; color: #0078d4;")
        layout.addWidget(title_label)

        sub_label = QLabel(
            f"Configuring PyTorch with CUDA acceleration for <b>{gpu_name}</b>.<br>"
            "This will download and install PyTorch and Torchvision (about 3 GB). "
            "Please keep your internet connection active."
            + ("<br><i>Installed into the 'ai_runtime' folder beside VisionView.exe; "
               "delete that folder to remove it.</i>" if getattr(sys, "frozen", False) else "")
        )
        sub_label.setWordWrap(True)
        sub_label.setStyleSheet("color: #444; font-size: 12px;")
        layout.addWidget(sub_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate busy indicator
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #e6e6e6;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status text
        self.status_label = QLabel("Starting installer...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #222;")
        layout.addWidget(self.status_label)

        # Expandable output log
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #dcdcdc;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.log_edit)

        # Button row
        btn_layout = QHBoxLayout()
        self.btn_toggle_log = QPushButton("Hide Details")
        self.btn_toggle_log.clicked.connect(self._toggle_log)
        btn_layout.addWidget(self.btn_toggle_log)

        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_install)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_close = QPushButton("Close")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

        # Worker thread
        self.worker = TorchInstallWorker(parent=self, driver_version=driver_version)
        self.worker.progress_output.connect(self._on_log)
        self.worker.status_changed.connect(self._on_status)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _toggle_log(self):
        if self.log_edit.isVisible():
            self.log_edit.hide()
            self.btn_toggle_log.setText("Show Details")
            self.resize(self.width(), 200)
        else:
            self.log_edit.show()
            self.btn_toggle_log.setText("Hide Details")
            self.resize(self.width(), 420)

    def _on_log(self, text):
        self.log_edit.appendPlainText(text)

    def _on_status(self, text):
        self.status_label.setText(text)

    def _on_finished(self, success, message):
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.hide()
        self.btn_close.setEnabled(True)
        self.btn_close.setText("Done")

        if success:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.status_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2e7d32;")
            # Load the new runtime into this session, then refresh the compatibility cache
            self.status_label.setText("Loading PyTorch into VisionView...")
            QApplication.processEvents()
            try:
                from utils.ai_runtime import activate_ai_runtime
                activate_ai_runtime()
            except Exception as e:
                self._on_log(f"[warning] Could not activate AI runtime: {e}")
            is_supported, reason, details = check_deep_learning_support(force_refresh=True)
            if is_supported:
                device = details.get("device", "CPU")
                message += f" AI features are ready ({device})."
            else:
                self._on_log(f"[info] {reason}")
                message += " Please restart VisionView to enable AI features."
            self.status_label.setText("✔ " + message)
            self.status_label.setWordWrap(True)
            # Reset module list cache if present
            try:
                import utils.process_canva as pc
                pc._CV_MODULES_CACHE = None
            except Exception:
                pass
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText("✖ " + message)
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #c62828;")
            if not self.log_edit.isVisible():
                self._toggle_log()

    def _cancel_install(self):
        reply = QMessageBox.question(
            self, "Cancel Installation",
            "Are you sure you want to cancel the PyTorch installation?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.status_label.setText("Cancelling installation...")
            self.worker.cancel()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self._cancel_install()
            if self.worker.isRunning():
                event.ignore()
                return
        event.accept()


def check_startup_ai_environment(parent=None, force_show=False):
    """
    Scans the system at startup for PyTorch environment and GPU compatibility.
    - If torch is not detected and device has an NVIDIA GPU:
        Prompts user to install Torch and Torchvision with CUDA support.
    - If device does not have an NVIDIA GPU:
        Prompts warning that GPU cannot be found.
    - Supports 'Don't ask again next time' checkbox on both popups.
    """
    if os.environ.get("VISIONVIEW_HEADLESS_TEST", "0") == "1":
        # In automated test mode, return structured diagnostic dictionary without modal blocking
        return {
            "status": "headless_bypass",
            "is_supported": check_deep_learning_support()[0],
            "gpu_info": detect_gpu_hardware()
        }

    is_supported, reason, details = check_deep_learning_support()

    # If PyTorch is already fully supported and CUDA is active:
    if is_supported and details.get("cuda"):
        if force_show:
            dev = details.get("device", "NVIDIA GPU")
            ver = details.get("torch_version", "")
            QMessageBox.information(
                parent, "AI Environment Ready",
                f"<b>Deep Learning is fully configured!</b><br><br>"
                f"PyTorch {ver} is installed with active CUDA acceleration on <b>{dev}</b>."
            )
        return {"status": "already_supported"}

    # PyTorch is either missing or lacks CUDA acceleration:
    gpu_info = detect_gpu_hardware()
    has_nvidia = gpu_info.get("has_nvidia", False)

    # -------------------------------------------------------------
    # CASE A: Device HAS an NVIDIA GPU, but Torch/Torchvision missing
    # -------------------------------------------------------------
    if has_nvidia:
        gpu_name = gpu_info["nvidia_gpus"][0] if gpu_info.get("nvidia_gpus") else "NVIDIA GPU"
        dont_ask = get_device_setting("startup/dont_ask_torch_setup", False)

        if not force_show and dont_ask:
            return {"status": "ignored_by_preference"}

        msg_box = QMessageBox(parent)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Torch & CUDA Environment Setup")
        msg_box.setText("<b>Torch and torchvision not detected</b>")
        msg_box.setInformativeText(
            f"An NVIDIA GPU was detected on this device: <b>{gpu_name}</b>.<br><br>"
            "PyTorch and Torchvision are required to run Deep Learning inference and AI inspection nodes.<br><br>"
            "Would you like to install PyTorch with CUDA acceleration now to enable all AI features?"
        )
        install_btn = msg_box.addButton("Install", QMessageBox.AcceptRole)
        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        msg_box.setDefaultButton(install_btn)

        cb = QCheckBox("Don't ask again next time")
        msg_box.setCheckBox(cb)

        msg_box.exec()

        if cb.isChecked():
            set_device_setting("startup/dont_ask_torch_setup", True)

        if msg_box.clickedButton() == install_btn:
            installer = TorchInstallDialog(parent=parent, gpu_name=gpu_name,
                                           driver_version=gpu_info.get("driver_version"))
            installer.exec()
            return {"status": "installer_launched"}
        return {"status": "user_cancelled"}

    # -------------------------------------------------------------
    # CASE B: Device DOES NOT have an NVIDIA GPU
    # -------------------------------------------------------------
    else:
        dont_ask = get_device_setting("startup/dont_ask_no_gpu_warning", False)
        if not force_show and dont_ask:
            return {"status": "ignored_by_preference"}

        msg_box = QMessageBox(parent)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("GPU Not Found")
        msg_box.setText("<b>GPU cannot be found</b>")
        msg_box.setInformativeText(
            "No compatible NVIDIA GPU was detected on this device.<br><br>"
            "Deep Learning and AI inspection modules require a dedicated NVIDIA GPU for real-time video processing.<br><br>"
            "<i>All standard inspection features (cameras, measurements, OpenCV filters, and video recording) will run normally without a GPU.</i>"
        )
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)

        cb = QCheckBox("Don't ask again next time")
        msg_box.setCheckBox(cb)

        msg_box.exec()

        if cb.isChecked():
            set_device_setting("startup/dont_ask_no_gpu_warning", True)

        return {"status": "no_gpu_warned"}


def show_unsupported_feature_popup(parent=None, feature_name="Deep Learning", reason=None):
    """
    Displays a user-friendly modal warning dialog when a user attempts
    to use a feature that is not supported by their laptop or system.
    """
    if reason is None:
        _, reason, _ = check_deep_learning_support()

    if os.environ.get("VISIONVIEW_HEADLESS_TEST", "0") == "1":
        return 1

    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Feature Not Supported")
    msg_box.setText("<b>This feature is not supported on your device.</b>")
    
    info_text = (
        f"The <b>{feature_name}</b> feature requires PyTorch and compatible hardware, "
        f"which is currently unavailable on this device.<br><br>"
        f"<b>Reason:</b><br>{reason}<br><br>"
        f"<i>All other VisionView features (camera streaming, OpenCV filters, measurement tools, and recording) "
        f"will continue to work normally.</i>"
    )
    msg_box.setInformativeText(info_text)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setDefaultButton(QMessageBox.Ok)
    return msg_box.exec()
