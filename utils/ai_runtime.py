# ==============================================================================
# VisionView - Side-by-side AI Runtime (PyTorch / Torchvision / RF-DETR)
#
# The frozen VisionView.exe has no Python interpreter or pip, so heavy AI
# libraries are installed into a separate environment *beside* the app:
#
#   VisionView-Windows-x64/
#       VisionView.exe
#       _internal/            (PyInstaller bundle, ships uv.exe)
#       ai_runtime/
#           env/              (venv: torch, torchvision, rfdetr, ...)
#           python/           (uv-managed CPython matching the app's version)
#           bin/uv.exe        (only if uv had to be downloaded)
#
# At startup activate_ai_runtime() appends env/site-packages (and the managed
# Python's stdlib as a fallback for modules PyInstaller did not bundle) to
# sys.path, so the existing `import torch` code works unchanged.
# Bundled modules (numpy, cv2, ...) always win because PyInstaller's frozen
# importer is consulted before sys.path; numpy is pinned to the bundled version
# so torch is installed against the exact numpy the app runs with.
#
# Keep this module free of heavy imports (no torch / cv2 / numpy at import time).
# ==============================================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path

UV_VERSION = "0.10.9"   # used only when uv must be downloaded at runtime
TORCH_INDEX_BASE = "https://download.pytorch.org/whl"
EXTRA_AI_PACKAGES = ["rfdetr", "supervision"]
MIN_FREE_GB = 6

IS_WINDOWS = sys.platform == "win32"
IS_FROZEN = bool(getattr(sys, "frozen", False))
_ACTIVATED_ENV = None


# ------------------------------------------------------------------------------
# Locations
# ------------------------------------------------------------------------------
def get_app_dir():
    """Folder that contains VisionView.exe (frozen) or the project root (source)."""
    if IS_FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _candidate_roots():
    roots = [get_app_dir() / "ai_runtime"]
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA")
        if base:
            roots.append(Path(base) / "VisionView" / "ai_runtime")
    else:
        roots.append(Path.home() / ".local" / "share" / "VisionView" / "ai_runtime")
    return roots


def _is_writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except Exception:
        return False


def get_install_root():
    """First writable runtime location (beside the exe, else per-user app data)."""
    for root in _candidate_roots():
        if _is_writable(root):
            return root
    return _candidate_roots()[-1]


def env_python(env_dir):
    env_dir = Path(env_dir)
    return env_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def env_site_packages(env_dir):
    env_dir = Path(env_dir)
    if IS_WINDOWS:
        return env_dir / "Lib" / "site-packages"
    return env_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def find_existing_env():
    """Return the env dir of an installed runtime, or None."""
    for root in _candidate_roots():
        env_dir = root / "env"
        if (env_dir / "pyvenv.cfg").exists() and env_site_packages(env_dir).exists():
            return env_dir
    return None


def _venv_home(env_dir):
    """Base interpreter folder recorded in pyvenv.cfg ('home = ...')."""
    try:
        for line in (Path(env_dir) / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() == "home":
                return Path(value.strip())
    except Exception:
        pass
    return None


def write_log(message):
    """Append a line to ai_runtime/ai_runtime.log (the frozen app has no console)."""
    try:
        import datetime
        env_dir = _ACTIVATED_ENV or find_existing_env()
        root = Path(env_dir).parent if env_dir else get_install_root()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(root / "ai_runtime.log", "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")
    except Exception:
        pass


def should_use_runtime():
    """The side-by-side runtime is used by the frozen app (or when forced for testing)."""
    return IS_FROZEN or os.environ.get("VISIONVIEW_USE_AI_RUNTIME", "0") == "1"


# ------------------------------------------------------------------------------
# Activation (called once at startup, before anything imports torch)
# ------------------------------------------------------------------------------
class _RuntimeSubmoduleFinder:
    """
    Last-resort import finder for the frozen app.
    PyInstaller bundles only the submodules VisionView itself uses, e.g. part of
    PIL or of the stdlib 'html' package. Such a bundled package shadows the
    complete copy, so `from PIL import ImageEnhance` or `import html.parser`
    (both needed by torchvision) would fail. When the normal finders cannot
    locate a submodule, look for it in the runtime's site-packages and in the
    runtime Python's stdlib.
    """

    def __init__(self, roots):
        self.roots = [Path(r) for r in roots]

    def find_spec(self, fullname, path=None, target=None):
        if "." not in fullname:
            return None
        import importlib.machinery
        parts = fullname.split(".")
        for root in self.roots:
            pkg_dir = root.joinpath(*parts[:-1])
            if pkg_dir.is_dir():
                spec = importlib.machinery.PathFinder.find_spec(fullname, [str(pkg_dir)])
                if spec is not None:
                    return spec
        return None


def activate_ai_runtime():
    """Make packages from the side-by-side runtime importable. Returns the env dir or None."""
    global _ACTIVATED_ENV
    if _ACTIVATED_ENV is not None:
        return _ACTIVATED_ENV
    if not should_use_runtime():
        return None
    env_dir = find_existing_env()
    if env_dir is None:
        return None

    import site
    site_dir = env_site_packages(env_dir)
    site.addsitedir(str(site_dir))  # appends to sys.path and processes .pth files

    dll_dirs = [site_dir / "torch" / "lib"]
    home = _venv_home(env_dir)
    stdlib_dirs = []
    if home is not None:
        if IS_WINDOWS:
            stdlib_dirs = [home / "Lib", home / "DLLs"]
            dll_dirs.append(home)  # vcruntime140_1.dll etc.
        else:
            stdlib_base = home.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            stdlib_dirs = [stdlib_base, stdlib_base / "lib-dynload"]
        # Fallback only: stdlib modules torch needs but PyInstaller did not bundle
        # (e.g. unittest). Appended last so bundled modules keep priority.
        for d in stdlib_dirs:
            if d.exists() and str(d) not in sys.path:
                sys.path.append(str(d))

    if IS_FROZEN:
        sys.meta_path.append(_RuntimeSubmoduleFinder(
            [site_dir] + [d for d in stdlib_dirs if d.exists()]))

    if IS_WINDOWS:
        for d in dll_dirs:
            if d.exists():
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    _ACTIVATED_ENV = env_dir
    write_log(f"Activated AI runtime: {env_dir} (app Python {sys.version.split()[0]}, "
              f"base home: {home}, frozen={IS_FROZEN})")
    return env_dir


# ------------------------------------------------------------------------------
# Install planning
# ------------------------------------------------------------------------------
def select_torch_index(driver_version):
    """
    Pick the PyTorch wheel index for the installed NVIDIA driver.
    Override with VISIONVIEW_TORCH_INDEX (e.g. 'cu126' or 'cpu').
    """
    override = os.environ.get("VISIONVIEW_TORCH_INDEX")
    if override:
        return override.strip()
    if not driver_version:
        return "cpu"
    try:
        major_minor = tuple(int(p) for p in str(driver_version).split(".")[:2])
    except ValueError:
        return "cu126"
    if major_minor >= (570, 0):   # CUDA 12.8 (required for RTX 50-series)
        return "cu128"
    if major_minor >= (528, 0):   # CUDA 12.6
        return "cu126"
    if major_minor >= (452, 0):   # CUDA 11.8
        return "cu118"
    return "cpu"


def torch_index_url(tag):
    return f"{TORCH_INDEX_BASE}/{tag}"


def bundled_numpy_version():
    try:
        import numpy
        return numpy.__version__
    except Exception:
        return None


def bundled_pillow_version():
    try:
        import PIL
        return PIL.__version__
    except Exception:
        return None


def find_uv():
    exe = "uv.exe" if IS_WINDOWS else "uv"
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / exe)
    app_dir = get_app_dir()
    candidates += [app_dir / exe, app_dir / "tools" / exe]
    candidates += [root / "bin" / exe for root in _candidate_roots()]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("uv")
    return Path(found) if found else None


def download_uv(dest_dir, log=print):
    """Download the standalone uv binary (fallback when it was not bundled)."""
    import platform
    import tarfile
    import tempfile
    import urllib.request
    import zipfile

    machine = platform.machine().lower()
    if IS_WINDOWS:
        asset = "uv-x86_64-pc-windows-msvc.zip"
    elif machine in ("aarch64", "arm64"):
        asset = "uv-aarch64-unknown-linux-gnu.tar.gz"
    else:
        asset = "uv-x86_64-unknown-linux-gnu.tar.gz"
    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}"
    log(f"Downloading uv {UV_VERSION} from {url}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    exe = "uv.exe" if IS_WINDOWS else "uv"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset
        urllib.request.urlretrieve(url, archive)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                member = next(n for n in zf.namelist() if n.endswith(exe))
                with zf.open(member) as src, open(dest_dir / exe, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            with tarfile.open(archive) as tf:
                member = next(m for m in tf.getmembers() if m.name.endswith("/" + exe) or m.name == exe)
                with tf.extractfile(member) as src, open(dest_dir / exe, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            os.chmod(dest_dir / exe, 0o755)
    return dest_dir / exe


class InstallStep:
    """One command in the install plan. `cmd` may be a callable returning the command list."""

    def __init__(self, status, cmd, optional=False):
        self.status = status
        self.cmd = cmd
        self.optional = optional

    def resolve(self):
        return self.cmd() if callable(self.cmd) else self.cmd


def uv_environment(root):
    """Keep everything uv downloads inside the runtime folder."""
    env = os.environ.copy()
    env["UV_PYTHON_INSTALL_DIR"] = str(Path(root) / "python")
    env["UV_CACHE_DIR"] = str(Path(root) / "cache")
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    env["UV_LINK_MODE"] = "copy"
    env["PYTHONUTF8"] = "1"
    return env


def build_runtime_plan(uv, root, index_tag, numpy_version):
    """Install plan for the side-by-side runtime used by the frozen app."""
    root = Path(root)
    env_dir = root / "env"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    index = torch_index_url(index_tag)
    numpy_pin = [f"numpy=={numpy_version}"] if numpy_version else ["numpy"]
    # Packages also bundled inside VisionView.exe are pinned to the bundled version,
    # because the bundled copy is the one that gets imported.
    pillow_version = bundled_pillow_version()
    shared_pins = numpy_pin + ([f"pillow=={pillow_version}"] if pillow_version else [])
    # NOTE: no --constraint files: uv splits --constraint values on spaces, which
    # breaks install folders like "...\computer vision\...". Version pins are
    # passed as plain requirements instead (already-installed pins are kept as-is).

    def numpy_cmd():
        return [str(uv), "pip", "install", "--python", str(env_dir), *numpy_pin]

    def torch_cmd():
        # NumPy is already installed at the bundled version; uv keeps it because it
        # satisfies torchvision's requirement, so only the PyTorch index is needed here.
        return [str(uv), "pip", "install", "--python", str(env_dir),
                "torch", "torchvision", "--index-url", index]

    def extras_cmd():
        # Pin the torch build just installed so RF-DETR cannot swap it for a CPU wheel.
        out = subprocess.check_output(
            [str(env_python(env_dir)), "-c",
             "import importlib.metadata as m; print(m.version('torch')); print(m.version('torchvision'))"],
            text=True, creationflags=_NO_WINDOW)
        torch_v, tv_v = out.split()
        return [str(uv), "pip", "install", "--python", str(env_dir), *EXTRA_AI_PACKAGES,
                f"torch=={torch_v}", f"torchvision=={tv_v}", *shared_pins,
                "--extra-index-url", index, "--index-strategy", "unsafe-best-match"]

    verify = [str(env_python(env_dir)), "-c",
              "import torch, torchvision; "
              "print('torch', torch.__version__, '| torchvision', torchvision.__version__, "
              "'| CUDA available:', torch.cuda.is_available())"]

    return [
        InstallStep(f"Preparing Python {py_ver} runtime...",
                    [str(uv), "venv", str(env_dir), "--python", py_ver, "--clear"]),
        InstallStep("Installing NumPy (matched to VisionView)...", numpy_cmd),
        InstallStep(f"Downloading PyTorch ({index_tag}) - large download, please wait...", torch_cmd),
        InstallStep("Installing RF-DETR and supervision...", extras_cmd, optional=True),
        InstallStep("Verifying PyTorch installation...", verify),
    ]


def build_source_plan(python_exe, index_tag, numpy_version):
    """Install plan when running from source: install into the current interpreter."""
    pins = [f"numpy=={numpy_version}"] if numpy_version else []
    return [
        InstallStep(f"Downloading PyTorch ({index_tag}) - large download, please wait...",
                    [str(python_exe), "-m", "pip", "install", "torch", "torchvision", *pins,
                     "--index-url", torch_index_url(index_tag),
                     "--extra-index-url", "https://pypi.org/simple",
                     "--no-warn-script-location"]),
    ]


_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


def free_space_gb(path):
    try:
        p = Path(path)
        while not p.exists():
            p = p.parent
        return shutil.disk_usage(p).free / (1024 ** 3)
    except Exception:
        return None
