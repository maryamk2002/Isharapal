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

set PYTHON=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please update the PYTHON path in this script.
    pause
    exit /b 1
)

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
