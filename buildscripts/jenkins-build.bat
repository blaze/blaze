REM This is the jenkins build script for building/testing
REM Blaze.
REM
REM Jenkins Requirements:
REM   - Anaconda should be installed in ~/anaconda
REM   - Use a jenkins build matrix for multiple
REM     platforms/python versions
REM   - Use the XShell plugin to launch this script
REM   - Call the script from the root workspace
REM     directory as buildscripts/jenkins-build
REM

REM Require a version of Python to be selected
if "%PYTHON_VERSION%" == "" exit /b 1

REM Jenkins has '/' in its workspace. Fix it to '\' to simplify the DOS commands.
set WORKSPACE=%WORKSPACE:/=\%

REM Use conda to create a conda environment of the required
REM python version and containing the dependencies.
SET PYENV_PREFIX=%WORKSPACE%\build\pyenv
rd /q /s %PYENV_PREFIX%
call C:\Anaconda\Scripts\conda create --yes -p %PYENV_PREFIX% python=%PYTHON_VERSION%  cython=0.19 scipy llvmpy ply
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on
set PYTHON_EXECUTABLE=%PYENV_PREFIX%\Python.exe
set PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

REM Build/install Blaze
%PYTHON_EXECUTABLE% setup.py install
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Run the tests
pushd ..
%PYTHON_EXECUTABLE% -c "import blaze;blaze.test(xunitfile='../test_results.xml', verbosity=2, exit=1)"
IF %ERRORLEVEL% NEQ 0 exit /b 1
popd

exit /b 0
