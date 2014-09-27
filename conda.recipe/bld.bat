
SET BLD_DIR=%CD%
cd /D "%RECIPE_DIR%\.."

"%PYTHON%" setup.py install

REM # X.X.X.dev builds
FOR /F "delims=" %%i IN ('git describe --tags') DO set BLAZE_VERSION=%%i
set _result=%ALPHA:-=_%

echo %_result%>__conda_version__.txt

copy __conda_version__.txt "%BLD_DIR%"

if errorlevel 1 exit 1

