@echo off
rem Cross-platform launcher for Windows cmd.exe.
rem
rem   modelbench.cmd setup                      create .venv and install deps
rem   modelbench.cmd web [--port 8765]          local web console
rem   modelbench.cmd run --models minicpm5-1b   CLI benchmark
rem
rem Locates the repository from this script (the script's own folder), never
rem from the current directory.  Python discovery for setup:
rem MB_PYTHON -> py -3.12 -> python.
rem Other commands always run the venv interpreter .venv\Scripts\python.exe;
rem arguments are forwarded verbatim and the exit code is propagated.
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 (
  echo modelbench: cannot enter repository root: %ROOT% 1>&2
  exit /b 1
)

set "CMD=%~1"
if "%CMD%"=="" goto :help
if /i "%CMD%"=="help" goto :help
if /i "%CMD%"=="--help" goto :help
if /i "%CMD%"=="-h" goto :help
if /i "%CMD%"=="setup" goto :setup

rem ----- normal CLI commands ----------------------------------------------
:run
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo environment not ready: run modelbench.cmd setup 1>&2
  popd >nul
  exit /b 2
)
"%VENV_PY%" -m modelbench.interfaces.cli %*
set "RC=%ERRORLEVEL%"
popd >nul
exit /b %RC%

:help
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo environment not ready: run modelbench.cmd setup 1>&2
  popd >nul
  exit /b 2
)
"%VENV_PY%" -m modelbench.interfaces.cli --help
set "RC=%ERRORLEVEL%"
popd >nul
exit /b %RC%

rem ----- setup ------------------------------------------------------------
:setup
if not exist "%ROOT%tool\bootstrap.py" (
  echo modelbench: missing %ROOT%tool\bootstrap.py 1>&2
  popd >nul
  exit /b 1
)
call :find_python
if errorlevel 1 (
  popd >nul
  exit /b 1
)
set "BOOT_ARGS="
:setup_collect
shift
if "%~1"=="" goto :setup_run
set "BOOT_ARGS=%BOOT_ARGS% "%~1""
goto :setup_collect
:setup_run
"%PY%" "%ROOT%tool\bootstrap.py" --venv "%ROOT%.venv" %BOOT_ARGS%
set "RC=%ERRORLEVEL%"
popd >nul
exit /b %RC%

rem ----- python discovery -------------------------------------------------
:find_python
set "PY="
if defined MB_PYTHON (
  "%MB_PYTHON%" -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    set "PY=%MB_PYTHON%"
    rem exit /b is safer than goto :eof inside a parenthesised block.
    exit /b 0
  )
  echo modelbench: MB_PYTHON=%MB_PYTHON% is not a runnable Python interpreter 1>&2
  exit /b 1
)
for /f "delims=" %%P in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY goto :eof
for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY goto :eof
echo modelbench: Python 3.10+ not found. Install Python (python.org) or set MB_PYTHON=C:\path\to\python.exe 1>&2
exit /b 1
