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

rd /q /s %WORKSPACE%\build
REM Fail if we couldn't fully remove the temporary dir
if EXIST %WORKSPACE%\build\ exit /b 1

REM Use conda to create a conda environment of the required
REM python version and containing the dependencies.
SET PYENV_PREFIX=%WORKSPACE%\build\pyenv
REM TODO: Add cffi to this list once it is added to anaconda windows.

call C:\Anaconda\Scripts\conda create --yes --channel https://conda.binstar.org/mwiebe -p %PYENV_PREFIX% python=%PYTHON_VERSION%  cython scipy ply dynd-python nose flask pyparsing pyyaml setuptools dateutil pip pytables sqlalchemy h5py pandas requests pytest toolz cytoolz bcolz pymongo psutil || exit /b 1


call C:\Anaconda\Scripts\conda install -p %PYENV_PREFIX% --yes --channel blaze mongodb || exit /b 1

echo on
set PYTHON_EXECUTABLE=%PYENV_PREFIX%\Python.exe
set PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

REM Temporary hack to install datashape
rd /q /s datashape
git clone https://github.com/ContinuumIO/datashape.git || exit /b 1
pushd datashape
%PYTHON_EXECUTABLE% setup.py install || exit /b 1
popd

REM bcolz needs unittest2
IF "%PYTHON_VERSION%" == "2.6" call pip install unittest2

call pip install multipledispatch

IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Build/install Blaze
%PYTHON_EXECUTABLE% setup.py install || exit /b 1

REM create a mongodb process
set dbpath=%PYENV_PREFIX%\mongodata\db
set logpath=%PYENV_PREFIX%\mongodata\log
mkdir %dbpath%
mkdir %logpath%

REM /b -> start without creating a new window. disables ^C handling
start /b mongod.exe --dbpath %dbpath% --logpath %logpath%\mongod.log || exit /b 1

call py.test --doctest-modules -vv --pyargs blaze --junitxml=test_results.xml

set testerror=%errorlevel%

REM /im -> process name, /f -> force kill
taskkill /im mongod.exe /f || exit /b 1

if %testerror% NEQ 0 exit /b 1

FOR /F "delims=" %%i IN ('git describe --tags --dirty --always --match [0-9]*') DO set BLAZE_VERSION=%%i
if "%BLAZE_VERSION%" == "" exit /b 1

REM Create a conda package from the build
call C:\Anaconda\Scripts\conda package -p %PYENV_PREFIX% --pkg-name=blaze --pkg-version=%BLAZE_VERSION%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on

REM Make sure binstar is installed in the main environment
echo Updating binstar...
call C:\Anaconda\Scripts\conda install --yes binstar || exit 1
call C:\Anaconda\Scripts\binstar --version

REM Upload the package to binstar
REM FOR /F "delims=" %%i IN ('dir /b blaze-*.tar.bz2') DO set PKG_FILE=%%i
REM call C:\Anaconda\Scripts\binstar -t %BINSTAR_BLAZE_AUTH% upload --force %PKG_FILE% || exit 1

cd ..

exit /b 0
