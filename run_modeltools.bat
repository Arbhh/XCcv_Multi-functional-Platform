@echo off
rem XCcv Model Tools launcher (ASCII only).
rem Interpreter precedence:
rem   1) XCCV_PY environment variable
rem   2) python_path.txt in this folder (one line: full path to python.exe)
rem   3) python on PATH
rem   4) py launcher (-3)
rem   5) .\venv and .\.venv in this folder (optional, last resort)
rem Each candidate is accepted only if it can import PyQt5.QtWebEngineWidgets.
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYEXE="
set "PYARGS="
set "PYCFG="
if exist "%ROOT%python_path.txt" set /p PYCFG=<"%ROOT%python_path.txt"

call :try "%XCCV_PY%" ""
call :try "%PYCFG%" ""
call :try "python" ""
call :try "py" "-3"
call :try "%ROOT%venv\Scripts\python.exe" ""
call :try "%ROOT%.venv\Scripts\python.exe" ""
if not defined PYEXE goto nogui

"%PYEXE%" %PYARGS% -m xccv.modeltools.modeltools_window

exit /b %errorlevel%

:try
if defined PYEXE exit /b 0
if "%~1"=="" exit /b 0
"%~1" %~2 -c "import PyQt5.QtWebEngineWidgets" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYEXE=%~1"
set "PYARGS=%~2"
exit /b 0

:nogui
echo.
echo [XCcv] No usable Python found (PyQtWebEngine is required).
echo         Create python_path.txt in this folder with one line, e.g.:
echo           C:\Python311\python.exe
echo         or set:  set XCCV_PY=C:\Python311\python.exe
echo.
if exist "%ROOT%tools\env_check.py" call :diag
pause
exit /b 1

:diag
python "%ROOT%tools\env_check.py" 2>nul
if errorlevel 1 echo (env_check.py needs a working python on PATH; skipped)
exit /b 0
