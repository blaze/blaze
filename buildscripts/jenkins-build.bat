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
call C:\Anaconda\Scripts\conda create --yes --channel http://repo.continuum.io/pkgs/dev -p %PYENV_PREFIX% python=%PYTHON_VERSION%  cython=0.19 scipy llvmpy ply numba dynd-python nose flask pyparsing pyyaml setuptools pip
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on
set PYTHON_EXECUTABLE=%PYENV_PREFIX%\Python.exe
set PATH=%PYENV_PREFIX%;%PYENV_PREFIX%\Scripts;%PATH%

REM Temporary hack to install pykit
rd /q /s pykit
git clone https://github.com/pykit/pykit.git
pushd pykit
%PYTHON_EXECUTABLE% setup.py install || exit /b 1
popd

REM Temporary hack to install datashape
rd /q /s datashape
git clone https://github.com/ContinuumIO/datashape.git
pushd datashape
%PYTHON_EXECUTABLE% setup.py install || exit /b 1
popd

REM Temporary hack to install blz
IF "%PYTHON_VERSION%" == "2.6" call pip install unittest2
rd /q /s blz
git clone https://github.com/ContinuumIO/blz.git
pushd blz
%PYTHON_EXECUTABLE% setup.py install || exit /b 1
popd

REM Build/install Blaze
%PYTHON_EXECUTABLE% setup.py install
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Run the tests (in a different directory, so the import works properly)
mkdir tmpdir
pushd tmpdir
%PYTHON_EXECUTABLE% -c "import blaze;blaze.test(xunitfile='../test_results.xml', verbosity=2, exit=1)"
IF %ERRORLEVEL% NEQ 0 exit /b 1
popd

exit /b 0
