@echo off
title Backend Server

REM --- 1. Setup Paths ---
REM This gets the directory where the .bat file is located
SET "SCRIPT_DIR=%~dp0"
REM Set paths for venv and requirements
SET "VENV_DIR=%SCRIPT_DIR%venv"
SET "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
SET "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
SET "REQS_FILE=%SCRIPT_DIR%requirements.txt"
SET "APP_FILE=%SCRIPT_DIR%app.py"


REM --- 2. Check if requirements.txt exists ---
if not exist "%REQS_FILE%" (
    echo "ERROR: requirements.txt not found!"
    echo "Please create a 'requirements.txt' file in this folder with:"
    echo "Flask"
    echo "flask-cors"
    pause
    exit /b
)

REM --- 3. Check if app.py exists ---
if not exist "%APP_FILE%" (
    echo "ERROR: app.py not found!"
    echo "Please make sure your server file is named 'app.py' and is in this folder."
    pause
    exit /b
)


REM --- 4. Check for 'venv' and Install Dependencies ---
if not exist "%VENV_DIR%" (
    echo "Virtual environment not found. Creating one..."
    
    REM Use the system's python to create the venv
    python -m venv "%VENV_DIR%"
    
    if %ERRORLEVEL% NEQ 0 (
        echo "ERROR: Failed to create virtual environment."
        echo "Please make sure Python 3 is installed and in your system PATH."
        pause
        exit /b
    )
    
    echo "Installing libraries from requirements.txt..."
    REM Use the pip from the new venv to install
    "%VENV_PIP%" install -r "%REQS_FILE%"
    
    if %ERRORLEVEL% NEQ 0 (
        echo "ERROR: Failed to install libraries."
        echo "Please check requirements.txt and your internet connection."
        pause
        exit /b
    )
    
    echo "Installation complete."
) else (
    echo "Virtual environment found. Skipping installation."
)


REM --- 5. Run the Server ---
echo.
echo ===================================
echo  Starting Flask server...
echo  (Press CTRL+C in this window to stop)
echo ===================================
echo.

REM Run the app using the venv's Python.
REM This runs it in the *current* window so you can see output.
"%VENV_PYTHON%" "%APP_FILE%"

echo.
echo Server has been stopped.
pause