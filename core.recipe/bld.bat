@echo off

conda remove markupsafe --yes
conda install markupsafe --yes
%PYTHON% setup.py --quiet install
