@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  set "PYTHON_CMD="
  set "PYTHON_ARGS="
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -c "import struct,sys; raise SystemExit(0 if sys.version_info >= (3,11) and sys.version_info < (3,14) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 (
      set "PYTHON_CMD=py"
      set "PYTHON_ARGS=-3.12"
    )
  )
  if not defined PYTHON_CMD (
    for %%C in (python python3) do (
      if not defined PYTHON_CMD (
        where %%C >nul 2>nul
        if not errorlevel 1 (
          %%C -c "import struct,sys; raise SystemExit(0 if sys.version_info >= (3,11) and sys.version_info < (3,14) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
          if not errorlevel 1 set "PYTHON_CMD=%%C"
        )
      )
    )
  )
  if not defined PYTHON_CMD (
    echo Could not find 64-bit Python 3.11, 3.12, or 3.13.
    echo Install Python from python.org, enable Add Python to PATH, then rerun.
    echo The py launcher is optional. You can check an installation with: python --version
    exit /b 1
  )
  call :create_venv
  if errorlevel 1 (
    echo Failed to create .venv with %PYTHON_CMD% %PYTHON_ARGS%.
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 exit /b 1
echo Setup complete. Run run_tests.bat, then run_demo.bat.
exit /b 0

:create_venv
"%PYTHON_CMD%" %PYTHON_ARGS% -m venv .venv
exit /b %errorlevel%
