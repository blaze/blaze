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

call C:\Anaconda\Scripts\conda create --yes --channel https://conda.binstar.org/mwiebe -p %PYENV_PREFIX% python=%PYTHON_VERSION%  cython scipy ply dynd-python nose flask pyparsing pyyaml setuptools dateutil pip pytables sqlalchemy h5py multipledispatch pandas requests || exit /b 1

echo on
set PYTHON_EXECUTABLE=%PYENV_PREFIX%\Python.exe
set PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

call pip install toolz cytoolz
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Temporary hack to install datashape
rd /q /s datashape
git clone https://github.com/ContinuumIO/datashape.git || exit /b 1
pushd datashape
%PYTHON_EXECUTABLE% setup.py install || exit /b 1
popd

REM Temporary hack to install blz
IF "%PYTHON_VERSION%" == "2.6" call pip install unittest2
rd /q /s blz
git clone https://github.com/ContinuumIO/blz.git || exit /b 1
pushd blz
%PYTHON_EXECUTABLE% setup.py install || exit /b 1
popd

REM Build/install Blaze
%PYTHON_EXECUTABLE% setup.py install || exit /b 1

call nosetests --with-doctest --exclude test_spark_\w* --with-xunit --xunit-file=test_results.xml || exit /b 1

exit /b 0
