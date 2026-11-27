@echo off
rem XCcv one-click environment installer (ASCII only).
rem Thin entry: locate a system Python 3 and hand over to tools\setup_env.py,
rem which performs GPU detection, torch index selection (cu128/cu126/cpu,
rem ultralytics 8.4.7 requires torch<2.10), venv creation, dependency
rem installation, python_path.txt and the final bilingual env check.
rem Interpreter probe order: XCCV_PY, py launcher (-3.11 then -3), python
rem on PATH. The dependency matrix is pinned to Python 3.11, so every
rem candidate must be Python 3.11 (newer Pythons lack wheels for pinned
rem packages like onnxruntime). The py launcher reads machine-registered
rem interpreters and is not affected by PATH shadowing (a wrong PATH
rem python would be picked first otherwise). The chosen interpreter is
rem echoed before launching.
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYEXE="
set "PYARGS="
call :try "%XCCV_PY%" ""
call :try "py" "-3.11"
call :try "py" "-3"
call :try "python" ""
if not defined PYEXE goto nopy

echo Using interpreter: %PYEXE% %PYARGS%
rem No --root is passed: setup_env.py derives the repo root from its own file
rem location (absolute, cwd-independent). Passing a trailing-backslash quoted
rem path like --root "%ROOT%" would corrupt argv under MSVCRT parsing (the
rem backslash escapes the closing quote and swallows the next arguments).
"%PYEXE%" %PYARGS% "%ROOT%tools\setup_env.py"
set "EXITCODE=%errorlevel%"
echo.
pause
exit /b %EXITCODE%

:try
if defined PYEXE exit /b 0
if "%~1"=="" exit /b 0
"%~1" %~2 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYEXE=%~1"
set "PYARGS=%~2"
exit /b 0

:nopy
echo.
echo [XCcv] No Python 3.11 found (the dependency matrix is pinned to Python 3.11).
echo         Install Python 3.11 from https://www.python.org/downloads/
echo         and enable "Add python.exe to PATH" during setup, then rerun.
echo         If 3.11 is already installed, verify with: py -3.11 --version
echo.
pause
exit /b 1
