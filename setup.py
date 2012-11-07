#!/usr/bin/env python
from distutils.core import setup

try:
    from nose.commands import nosetests
    testrunner = {'test': nosetests}
except:
    testrunner = {}

longdesc = open('README.md').read()

setup(
    name='blaze',
    version='dev',
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='Blaze',
    long_description=longdesc,
    data_files=[],
    entry_points={},
    cmdclass=testrunner,
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
    packages=['ndtable'],
    install_requires=[''],
)
