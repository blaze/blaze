#!/usr/bin/env python
from setuptools import setup, find_packages
from nose.commands import nosetests

longdesc = open('README.md').read()

setup(
    name='blaze',
    version='dev',
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='Blaze',
    long_description=longdesc,
    packages=find_packages(),
    install_requires=[''],
    data_files=[],
    entry_points={},
    cmdclass={'test': nosetests},
    license='BSD',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
    ],
)
