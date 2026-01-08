@echo off
REM ============================================================
REM PSL Recognition System V2 - Start Server
REM ============================================================

echo.
echo ============================================================
echo PSL RECOGNITION SYSTEM V2
echo Complete Urdu Alphabet Support (40 Signs)
echo ============================================================
echo.

cd /d "%~dp0backend"

REM Auto-detect Python installation
REM First try 'python' in PATH, then 'py' launcher, then common locations
set PYTHON=

REM Check if python is in PATH
where python >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python
    goto :found_python
)

REM Check if py launcher is available
where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py -3
    goto :found_python
)

REM Check common Python installation paths
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :found_python
)
if exist "C:\Python312\python.exe" (
    set "PYTHON=C:\Python312\python.exe"
    goto :found_python
)
if exist "C:\Python311\python.exe" (
    set "PYTHON=C:\Python311\python.exe"
    goto :found_python
)

REM Python not found
echo ERROR: Python not found!
echo.
echo Please install Python 3.10 or higher from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:found_python
echo Found Python: %PYTHON%

echo Python: %PYTHON%
echo.
echo Starting PSL V2 Server...
echo.
echo Access the application at:
echo   V2 Interface: http://localhost:5000/index_v2.html
echo   V1 Interface: http://localhost:5000/
echo.
echo Press Ctrl+C to stop the server.
echo.
echo ============================================================
echo.

%PYTHON% app_v2.py

pause
