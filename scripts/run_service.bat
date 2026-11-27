@echo off
rem XCcv inference service launcher (headless, ASCII only).
rem Interpreter precedence:
rem   1) XCCV_PY environment variable
rem   2) python_path.txt in the repo root (one line: full path to python.exe)
rem   3) python on PATH
rem   4) py launcher (-3)
rem   5) venv and .venv under the repo root (optional, last resort)
rem Each candidate is accepted only if it can import fastapi and uvicorn.
rem Extra arguments are passed through, e.g.:  run_service.bat --port 8600
setlocal EnableExtensions
set "ROOT=%~dp0.."
cd /d "%ROOT%"

set "PYEXE="
set "PYARGS="
set "PYCFG="
if exist "%ROOT%\python_path.txt" set /p PYCFG=<"%ROOT%\python_path.txt"

call :try "%XCCV_PY%" ""
call :try "%PYCFG%" ""
call :try "python" ""
call :try "py" "-3"
call :try "%ROOT%\venv\Scripts\python.exe" ""
call :try "%ROOT%\.venv\Scripts\python.exe" ""
if not defined PYEXE goto nopy

"%PYEXE%" %PYARGS% -m xccv.service.main %*

exit /b %errorlevel%

:try
if defined PYEXE exit /b 0
if "%~1"=="" exit /b 0
"%~1" %~2 -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYEXE=%~1"
set "PYARGS=%~2"
exit /b 0

:nopy
echo.
echo [XCcv] No usable Python found (fastapi and uvicorn are required).
echo         Create python_path.txt in the repo root with one line, e.g.:
echo           C:\Python311\python.exe
echo         or set:  set XCCV_PY=C:\Python311\python.exe
echo.
pause
exit /b 1
