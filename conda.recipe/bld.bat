
cd "%RECIPE_DIR%\.."
"%PYTHON%" setup.py build install
if errorlevel 1 exit 1

