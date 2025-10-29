@echo off
echo Starting the PyInstaller build process...

REM --- 1. Setup Paths ---
REM Get the directory where this .bat file is located
SET "SCRIPT_DIR=%~dp0"
REM Set path to the venv activate script
SET "VENV_ACTIVATE=%SCRIPT_DIR%.venv\Scripts\activate.bat"
REM Set path to the main Python script
SET "MAIN_SCRIPT=%SCRIPT_DIR%main.py"

REM --- 2. Check if venv activate script exists ---
if not exist "%VENV_ACTIVATE%" (
    echo "ERROR: Could not find the venv activate script."
    echo "Looked for it at: %VENV_ACTIVATE%"
    echo "Make sure your .venv folder is in the same directory as this .bat file."
    pause
    exit /b
)

REM --- 3. Check if main.py exists ---
if not exist "%MAIN_SCRIPT%" (
    echo "ERROR: Could not find main.py."
    echo "Looked for it at: %MAIN_SCRIPT%"
    pause
    exit /b
)

echo Activating virtual environment...
REM Call the activate script. This modifies the *current* command prompt session.
call "%VENV_ACTIVATE%"

echo Running PyInstaller...
REM Execute the PyInstaller command
pyinstaller --onefile --name emotion-gesture-tracker --add-data "face_landmarker.task;." --add-data "hand_landmarker.task;." --add-data "pictures;pictures" "%MAIN_SCRIPT%"

REM Check if PyInstaller was successful
if %ERRORLEVEL% NEQ 0 (
    echo "ERROR: PyInstaller failed. See the output above for details."
    pause
) else (
    echo "Build successful! The .exe file is in the 'dist' folder."
)

echo Deactivating virtual environment (optional, prompt closes anyway)...
REM Deactivate isn't strictly necessary as the script ends, but good practice
call deactivate

echo Build process finished.
pause REM Keep the window open to see the final messages