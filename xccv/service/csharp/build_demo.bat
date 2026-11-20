@echo off
rem Build XCcv C# demo (XCcvCSharpDemo.exe) with the .NET Framework in-box
rem csc.exe. The deploy platform card runs the same build automatically.
rem Usage: run this file from its folder (xccv\service\csharp).

setlocal
set OUT=%~dp0..\..\..\platform_data\csharp_demo
if not exist "%OUT%" mkdir "%OUT%"

set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
if not exist "%CSC%" (
  echo .NET Framework compiler csc.exe not found
  exit /b 1
)

"%CSC%" /nologo /target:winexe /out:"%OUT%\XCcvCSharpDemo.exe" ^
  /win32icon:"%~dp0..\..\assets\csharp_demo.ico" ^
  /r:System.Web.Extensions.dll /r:System.Net.Http.dll ^
  /r:System.dll /r:System.Core.dll /r:System.Drawing.dll /r:System.Windows.Forms.dll ^
  "%~dp0XCcv.Vision.Sdk.cs" "%~dp0Demo\XCcvDemo.cs"
if errorlevel 1 (
  echo Build FAILED
  exit /b 1
)
copy /y "%~dp0..\..\assets\csharp_demo.ico" "%OUT%\XCcvDemo.ico" >nul
echo Build OK: %OUT%\XCcvCSharpDemo.exe
endlocal
