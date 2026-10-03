@echo off
setlocal EnableExtensions

rem ============================================================================
rem  Laya Decision Service -- start it and open the Swagger UI.
rem
rem  Double-click this file, or run it from a terminal. It starts the service in
rem  its own window, waits until /health actually answers, and only then opens
rem  http://127.0.0.1:8000/docs in the default browser.
rem
rem  That wait matters. Opening the browser immediately would show "This site
rem  can't be reached" on a cold cache, because the server has not bound the port
rem  yet -- building the checkpoints takes seconds, and a first run that downloads
rem  them takes minutes.
rem
rem  Close the service window to stop it.
rem
rem  Override with environment variables, e.g.  set LAYA_PORT=9000 && start.cmd
rem ============================================================================

title Laya Decision Service

rem ------------------------------------------------------------------ configure
set "HOST=127.0.0.1"
set "PORT=8000"
rem How long to wait for /health before giving up and explaining what to check.
set "READY_TIMEOUT=180"

if not "%LAYA_HOST%"=="" set "HOST=%LAYA_HOST%"
if not "%LAYA_PORT%"=="" set "PORT=%LAYA_PORT%"

rem `transformers` probes for TensorFlow at import time, and when TensorFlow is
rem installed its abseil runtime can deadlock model construction: the process
rem hangs at load with no error at all. Set before laya is imported, which is why
rem it belongs here and not only in the Python entry point.
if "%USE_TF%"=="" set "USE_TF=0"

cd /d "%~dp0"

set "BASE=http://%HOST%:%PORT%"
set "DOCS=%BASE%/docs"

echo.
echo   Laya Decision Service
echo   ==========================================================
echo   Swagger   %DOCS%
echo   OpenAPI   %BASE%/openapi.json
echo   Health    %BASE%/health
echo   Metrics   %BASE%/metrics
echo.

rem ------------------------------------------------------------ python lookup
where python >nul 2>&1
if errorlevel 1 (
    echo   [X] 'python' was not found on PATH.
    echo       Install Python 3.10+ and re-run, or run this from an activated
    echo       virtual environment.
    goto :done
)

rem ------------------------------------------------------- already-running guard
rem If the port already serves, do not start a second copy: the second would fail
rem to bind while this script reported a successful start.
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo   [!] Port %PORT% is already serving. Opening Swagger against it.
    start "" "%DOCS%"
    goto :done
)

rem ------------------------------------------------------------------- launch
rem The child inherits this script's working directory, which is already the
rem project root, so no `cd` and no nested quoting are needed inside the command.
echo   Starting the service in a new window...
echo   A cold cache downloads the checkpoints first, so the first start is slow.
echo.

start "Laya Decision Service" cmd /k python -m app --host %HOST% --port %PORT%

rem --------------------------------------------------------------------- wait
rem One PowerShell call polls rather than a batch loop, so it can use a real
rem deadline. Its output is discarded; the echo below reports the outcome, and
rem PowerShell diagnostics are dropped because a failure here means "not ready
rem yet", which the timeout message explains better.
powershell -NoProfile -Command "$deadline = (Get-Date).AddSeconds(%READY_TIMEOUT%); while ((Get-Date) -lt $deadline) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%BASE%/health' -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } } catch { }; Start-Sleep -Seconds 2 }; exit 1" >nul 2>&1

if errorlevel 1 goto :timeout

echo   Ready. Opening Swagger...
start "" "%DOCS%"
goto :done

rem ----------------------------------------------------------------- timeouts
:timeout
echo.
echo   [X] /health did not answer within %READY_TIMEOUT%s.
echo.
echo   Check the service window for the reason. The usual causes:
echo     - A checkpoint is still downloading. Run  python -m scripts.warmup  and
echo       watch it finish, then start this again.
echo     - HF_HUB_OFFLINE=1 is set but the cache is incomplete. Re-run
echo       python -m scripts.warmup --models all --verify-offline  on a
echo       networked machine and ship the whole HF_HOME directory.
echo     - Port %PORT% is unusable. Try  set LAYA_PORT=9000
echo.
echo   Opening the address anyway, in case it is slow rather than broken.
start "" "%DOCS%"

:done
echo.
endlocal
