@echo off
rem XCcv Inference Service - NSSM registration template.
rem
rem Purpose: register the inference service as a Windows service so it
rem auto-starts on boot and restarts on crash.
rem
rem Usage (target machine, admin cmd):
rem   1) install NSSM (https://nssm.cc) and make nssm.exe reachable
rem      (put it on PATH or set NSSM to its full path below)
rem   2) run this file as administrator, then: sc query XCcvInferenceService
rem
rem Defaults: run the service with this machine's python and this repo folder.
rem If you ship a packaged launcher, set SVC_EXE to that exe instead.

setlocal EnableExtensions
set "ROOT=%~dp0..\.."
pushd "%ROOT%" && set "ROOT=%CD%" && popd

set "NSSM=nssm.exe"
set "APPDIR=%ROOT%"
rem Service executable: python.exe by default; override with a packaged exe if any
if not defined PYEXE set "PYEXE=python.exe"
set "SVC_EXE=%PYEXE%"
set "SVC_ARGS=-m xccv.service.main --host 127.0.0.1 --port 8600"

if not exist "%APPDIR%\xccv\service\main.py" (
  echo [XCcv] Service sources not found under %APPDIR%
  exit /b 1
)

"%NSSM%" install XCcvInferenceService "%SVC_EXE%"
"%NSSM%" set XCcvInferenceService AppParameters "%SVC_ARGS%"
"%NSSM%" set XCcvInferenceService AppDirectory "%APPDIR%"
"%NSSM%" set XCcvInferenceService AppStdout "%APPDIR%\platform_data\logs\service.out.log"
"%NSSM%" set XCcvInferenceService AppStderr "%APPDIR%\platform_data\logs\service.err.log"
"%NSSM%" set XCcvInferenceService AppRotateFiles 1
"%NSSM%" set XCcvInferenceService AppRotateOnline 1
"%NSSM%" set XCcvInferenceService AppRotateBytes 10485760
"%NSSM%" set XCcvInferenceService Start SERVICE_AUTO_START
"%NSSM%" start XCcvInferenceService
echo Registered and started: XCcvInferenceService
endlocal
