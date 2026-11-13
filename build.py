#!/usr/bin/env python3
"""
==============================================================================
VisionView - Universal Cross-Platform Build & Packaging Script
Supports: Windows (x86_64), Linux (x86_64), Linux ARM (aarch64 / armv7l)
==============================================================================
Usage:
    python build.py
    python build.py --clean
    python build.py --archive
==============================================================================
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"
SPEC_FILE = BASE_DIR / "VisionView.spec"

def get_platform_info():
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if "windows" in system:
        os_tag = "windows"
        arch_tag = "x64" if machine in ["amd64", "x86_64"] else machine
        exe_name = "VisionView.exe"
    elif "linux" in system:
        os_tag = "linux"
        if machine in ["aarch64", "arm64"]:
            arch_tag = "arm64"
        elif machine.startswith("arm"):
            arch_tag = "armv7l"
        elif machine in ["x86_64", "amd64"]:
            arch_tag = "x86_64"
        else:
            arch_tag = machine
        exe_name = "VisionView"
    elif "darwin" in system:
        os_tag = "macos"
        arch_tag = "arm64" if "arm" in machine else "x86_64"
        exe_name = "VisionView"
    else:
        os_tag = system
        arch_tag = machine
        exe_name = "VisionView"
        
    return os_tag, arch_tag, exe_name

def check_dependencies():
    print("[*] Checking build dependencies...")
    try:
        import PyInstaller
        print(f"  [+] PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("  [!] PyInstaller not found! Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.5.0"])
        
    core_pkgs = ["PySide6", "cv2", "numpy", "qt_material_icons", "pypylon", "pytesseract"]
    missing_pkgs = []
    for pkg in core_pkgs:
        try:
            __import__(pkg)
            print(f"  [+] {pkg}: Found")
        except ImportError:
            print(f"  [!] Missing required package: '{pkg}'")
            missing_pkgs.append(pkg)
            
    if missing_pkgs:
        print(f"\n[FATAL ERROR] Cannot build VisionView: missing core dependencies: {missing_pkgs}")
        print("Please ensure your environment has installed all requirements:")
        print("    pip install -r requirements.txt\n")
        sys.exit(1)

def clean_build_artifacts():
    print("[*] Cleaning build artifacts...")
    for p in [BUILD_DIR, DIST_DIR]:
        if p.exists():
            print(f"  [-] Removing {p}")
            shutil.rmtree(p, ignore_errors=True)

def run_build(clean=False):
    os_tag, arch_tag, exe_name = get_platform_info()
    
    print("=" * 70)
    print(f" VISIONVIEW STANDALONE COMPILER")
    print(f" Target OS:        {os_tag.upper()}")
    print(f" Architecture:     {arch_tag}")
    print(f" Python Version:   {platform.python_version()}")
    print(f" Executable Name:  {exe_name}")
    print("=" * 70)
    
    check_dependencies()
    
    if clean:
        clean_build_artifacts()
        
    if not SPEC_FILE.exists():
        print(f"[!] Error: Spec file not found at {SPEC_FILE}")
        sys.exit(1)
        
    print(f"\n[*] Running PyInstaller with spec: {SPEC_FILE.name}...")
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        str(SPEC_FILE)
    ]
    
    if clean:
        cmd.append("--clean")
        
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print("\n[!] Build failed with exit code:", result.returncode)
        sys.exit(result.returncode)
        
    out_dir = DIST_DIR / "VisionView"
    out_exe = out_dir / exe_name
    
    if not out_exe.exists():
        print(f"\n[!] Build completed but expected executable was not found: {out_exe}")
        sys.exit(1)
        
    # Ensure execution permissions on Linux/ARM
    if os_tag != "windows":
        out_exe.chmod(0o755)
        
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Standalone Application compiled successfully!")
    print(f"  Location: {out_dir}")
    print(f"  Executable: {out_exe}")
    print("=" * 70)
    
    return out_dir, os_tag, arch_tag

def create_archive(out_dir, os_tag, arch_tag):
    archive_base = f"VisionView-{os_tag}-{arch_tag}"
    print(f"\n[*] Packaging distribution archive: {archive_base}...")
    
    if os_tag == "windows":
        archive_path = DIST_DIR / f"{archive_base}.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for f in files:
                    file_path = Path(root) / f
                    arcname = file_path.relative_to(out_dir.parent)
                    zf.write(file_path, arcname)
        print(f"  [+] Created ZIP: {archive_path} ({archive_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        archive_path = DIST_DIR / f"{archive_base}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(out_dir, arcname="VisionView")
        print(f"  [+] Created TAR.GZ: {archive_path} ({archive_path.stat().st_size / (1024*1024):.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Build and package VisionView standalone application.")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before compiling.")
    parser.add_argument("--archive", action="store_true", default=True, help="Create portable distribution archive (zip/tar.gz).")
    parser.add_argument("--no-archive", dest="archive", action="store_false", help="Skip creating distribution archive.")
    args = parser.parse_args()
    
    out_dir, os_tag, arch_tag = run_build(clean=args.clean)
    if args.archive:
        create_archive(out_dir, os_tag, arch_tag)

if __name__ == "__main__":
    main()
