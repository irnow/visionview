@echo off
setlocal enabledelayedexpansion
title VisionView Automated Standalone Compiler

REM Ensure script runs in its own directory when double-clicked
cd /d "%~dp0"

echo ==========================================================
echo  VisionView Automated Standalone Compiler for Windows
echo ==========================================================

REM 1. Locate any working Python 3 interpreter on the host system
set PYTHON_EXE=

REM Check python in PATH (filtering out Windows Store stub)
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PYTHON_EXE (
        echo "%%i" | findstr /i "WindowsApps" >nul
        if errorlevel 1 (
            "%%i" -c "import sys" >nul 2>nul
            if not errorlevel 1 (
                set "PYTHON_EXE=%%i"
            )
        )
    )
)

REM Check Windows Python launcher (py)
if not defined PYTHON_EXE (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        py -3 -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
                set "PYTHON_EXE=%%i"
            )
        )
    )
)

REM Check common Python install locations
if not defined PYTHON_EXE (
    for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*" "C:\Program Files\Python3*" "C:\Python3*" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3") do (
        if not defined PYTHON_EXE (
            if exist "%%d\python.exe" (
                "%%d\python.exe" -c "import sys" >nul 2>nul
                if not errorlevel 1 (
                    set "PYTHON_EXE=%%d\python.exe"
                )
            )
        )
    )
)

if defined PYTHON_EXE (
    echo [+] Detected System Python: !PYTHON_EXE!
)

REM 2. If Python is still not found, automatically install via winget
if not defined PYTHON_EXE (
    echo [!] Python was not detected on this machine.
    echo [*] Automatically installing Python 3.10 via Windows Package Manager [winget]...
    where winget >nul 2>nul
    if !errorlevel! equ 0 (
        winget install --id Python.Python.3.10 -e --accept-source-agreements --accept-package-agreements
        for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
            if exist "%%d\python.exe" (
                set "PYTHON_EXE=%%d\python.exe"
            )
        )
    )
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] Python 3.10+ is required to build VisionView.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM 3. Automated Virtual Environment & Dependency Installation
if not exist ".build_venv\Scripts\python.exe" (
    echo [*] Creating isolated build virtual environment [.build_venv]...
    "!PYTHON_EXE!" -m venv .build_venv
    if not exist ".build_venv\Scripts\python.exe" (
        echo [ERROR] Failed to create virtual environment with !PYTHON_EXE!.
        pause
        exit /b 1
    )
)

set "VENV_PYTHON=.build_venv\Scripts\python.exe"

echo [*] Automatically checking and downloading required packages from requirements.txt...
"!VENV_PYTHON!" -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo [WARNING] pip upgrade failed, continuing with current pip...
)

"!VENV_PYTHON!" -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)

"!VENV_PYTHON!" -m pip install pyinstaller
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Failed to install pyinstaller.
    pause
    exit /b 1
)

REM 4. Execute compilation
echo.
echo [*] Compiling VisionView into standalone application...
"!VENV_PYTHON!" build.py %*

if !errorlevel! equ 0 (
    echo.
    echo ==========================================================
    echo [SUCCESS] VisionView built successfully!
    echo Application folder: dist\VisionView\
    echo Distribution ZIP:  dist\VisionView-windows-x64.zip
    echo ==========================================================
) else (
    echo.
    echo [ERROR] Build failed with exit code !errorlevel!.
)

pause
