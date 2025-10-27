@echo off

REM --- 1. Setup Paths ---
REM This gets the directory where the .bat file is located
SET "SCRIPT_DIR=%~dp0"
REM Set paths for venv and requirements
SET "VENV_DIR=%SCRIPT_DIR%.venv"
SET "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
SET "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
SET "REQS_FILE=%SCRIPT_DIR%requirements.txt"


REM --- 2. Check if requirements.txt exists ---
if not exist "%REQS_FILE%" (
    echo "ERROR: requirements.txt not found!"
    echo "This file is necessary to install the libraries."
    echo "Please create it first by running this in your terminal:"
    echo "pip freeze > requirements.txt"
    pause
    exit /b
)


REM --- 3. Check for .venv and Install Dependencies ---
if not exist "%VENV_DIR%" (
    echo "Virtual environment not found. Creating one..."
    
    REM Use the system's python to create the venv
    python -m venv .venv
    
    if %ERRORLEVEL% NEQ 0 (
        echo "ERROR: Failed to create virtual environment."
        echo "Please make sure Python 3.10 is installed and in your system PATH."
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


REM --- 4. Run the Scripts ---
echo "Starting scripts. Output will be saved to log files."

REM Overwrite log file (>) for the server
START "Emotion Display Server" cmd /c ""%VENV_PYTHON%" "display_emotion.py" > "display_server.log" 2>&1"

echo "Waiting 3 seconds for the server to start..."
timeout /t 3 /nobreak > nul

REM Overwrite log file (>) for the tracker
START "Emotion Tracker" cmd /c ""%VENV_PYTHON%" "tracker.py" > "tracker.log" 2>&1"

echo "Both scripts are running in the background."
echo "Your OpenCV windows (the camera and emotion) should appear."
echo "Close the OpenCV windows (press 'q') to stop the scripts."

REM This .bat window will close.