# -*- mode: python ; coding: utf-8 -*-
# ==============================================================================
# PyInstaller Build Specification for VisionView
# Supports: Windows (x86_64), Linux (x86_64, aarch64/ARM64)
# ==============================================================================

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, copy_metadata

block_cipher = None
base_dir = os.path.abspath(os.getcwd())

# 1. Base Project Data Folders
datas = [
    (os.path.join(base_dir, 'icons'), 'icons'),
    (os.path.join(base_dir, 'computer_vision'), 'computer_vision'),
    (os.path.join(base_dir, 'camera'), 'camera'),
    (os.path.join(base_dir, 'ai_modules'), 'ai_modules'),
]

binaries = []
hiddenimports = [
    # Core packages
    'cv2',
    'numpy',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'qt_material_icons',
    'pytesseract',
    
    # Internal project modules
    'camera.basler',
    'camera.camera_parameters',
    'camera.camera_worker',
    'camera.dialog',
    'camera.image_worker',
    'computer_vision.canny_edge_detection',
    'computer_vision.clahe',
    'computer_vision.crop',
    'computer_vision.deep_learning',
    'computer_vision.gaussian_blur',
    'computer_vision.parameter_dialog',
    'computer_vision.resize',
    'computer_vision.threshold',
    'computer_vision.color_segmentation',
    'computer_vision.morphology',
    'computer_vision.contour_analysis',
    'computer_vision.template_matching',
    'computer_vision.optical_character_recognition',
    'utils.annotators',
    'utils.calibration_dialog',
    'utils.directory_viewer',
    'utils.device_compatibility',
    'utils.gui_signal',
    'utils.metrology',
    'utils.process_canva',
    'utils.startup_splash',
    'utils.ai_runtime',
    'utils.video_label',
    'utils.window_module',
    'ai_modules.rfdetr_bbox',
    'ai_modules.rfdetr_segmentation',
]

# 2. Add all 12 qt_material_icons resource combinations
for style in ['outlined', 'rounded', 'sharp']:
    for size in [20, 24, 40, 48]:
        hiddenimports.append(f'qt_material_icons.resources.icons_{style}_{size}')

# 3. Collect Pypylon (Basler SDK)
try:
    pypylon_datas, pypylon_binaries, pypylon_hidden = collect_all('pypylon')
    datas += pypylon_datas
    binaries += pypylon_binaries
    hiddenimports += pypylon_hidden
except Exception as e:
    print(f"[SPEC WARNING] Could not collect all pypylon: {e}")

# 4. Collect OpenCV data/metadata
try:
    cv2_datas, cv2_binaries, cv2_hidden = collect_all('cv2')
    datas += cv2_datas
    binaries += cv2_binaries
    hiddenimports += cv2_hidden
except Exception:
    pass

# 5. Conditionally include AI / Deep Learning modules if installed
try:
    import rfdetr
    datas += copy_metadata('rfdetr')
    hiddenimports += ['rfdetr', 'rfdetr.config', 'ai_modules.rfdetr_bbox', 'ai_modules.rfdetr_segmentation']
except ImportError:
    pass

try:
    import supervision
    datas += copy_metadata('supervision')
    hiddenimports += ['supervision']
except ImportError:
    pass

# 5a. Bundle Pillow completely: torchvision (from ai_runtime/) imports PIL
#     submodules VisionView itself never uses (ImageEnhance, ImageOps, ...), and
#     the bundled PIL package shadows the runtime's copy.
from PyInstaller.utils.hooks import collect_submodules
hiddenimports += collect_submodules('PIL')

# 5b. Bundle uv (package installer) so the app can build its side-by-side AI
#     runtime (torch/torchvision) on the user's machine. See utils/ai_runtime.py.
import shutil
uv_exe = os.path.join(base_dir, 'tools', 'uv.exe' if sys.platform.startswith('win') else 'uv')
if not os.path.exists(uv_exe):
    uv_exe = shutil.which('uv')
if uv_exe:
    binaries.append((uv_exe, '.'))
    print(f"[SPEC] Bundling uv from: {uv_exe}")
else:
    print("[SPEC WARNING] uv not found; the app will download it on first AI install.")

# 6. Exclude bulky training/scientific packages not required at runtime
excludes = [
    'tkinter',
    'matplotlib',
    'scipy',
    'pandas',
    'clearml',
    'mlflow',
    'wandb',
    'tensorboard',
    'anomalib',
    'roboflow',
    'IPython',
    'notebook',
    'jupyter',
    'pytest',
    'unittest',
    'fastapi',
    'uvicorn',
    'Flask',
    'waitress',
    'altair',
    'botocore',
    'boto3',
    'databricks',
]

# Set platform-specific application icon
icon_file = None
if sys.platform.startswith('win'):
    ico_candidate = os.path.join(base_dir, 'icons', 'app_icon.ico')
    if os.path.exists(ico_candidate):
        icon_file = ico_candidate
else:
    png_candidate = os.path.join(base_dir, 'icons', 'process.png')
    if os.path.exists(png_candidate):
        icon_file = png_candidate

a = Analysis(
    [os.path.join(base_dir, 'main.py')],
    pathex=[base_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 7. Windows: ship a current Microsoft Visual C++ runtime.
#    The build Python drags in an old MSVCP140/VCRUNTIME140 (e.g. 14.27 from 2020).
#    PyTorch 2.x (installed later into ai_runtime/) fails with
#    "WinError 1114 ... c10.dll" against such an old runtime, so replace the
#    top-level copies with the build machine's System32 versions (VC++ redist)
#    and add the newer runtime DLLs torch needs, for PCs without a recent redist.
if sys.platform.startswith('win'):
    _sys32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    _vc_runtime = {
        'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll', 'msvcp140_atomic_wait.dll',
        'msvcp140_codecvt_ids.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
        'vcruntime140_threads.dll', 'concrt140.dll',
    }
    _replaced, _new_binaries = set(), []
    for _entry in a.binaries:
        _dest, _src, _kind = _entry[0], _entry[1], _entry[2]
        _name = os.path.basename(_dest).lower()
        if os.path.dirname(_dest) == '' and _name in _vc_runtime:
            _sys_copy = os.path.join(_sys32, _name)
            if os.path.exists(_sys_copy):
                _new_binaries.append((_dest, _sys_copy, _kind))
                _replaced.add(_name)
                continue
        _new_binaries.append(_entry)
    for _name in sorted(_vc_runtime - _replaced):
        _sys_copy = os.path.join(_sys32, _name)
        if os.path.exists(_sys_copy):
            _new_binaries.append((_name, _sys_copy, 'BINARY'))
            _replaced.add(_name)
    a.binaries = _new_binaries
    print(f"[SPEC] Using System32 MSVC runtime for: {', '.join(sorted(_replaced))}")

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionView',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VisionView',
)
