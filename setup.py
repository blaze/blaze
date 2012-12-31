#!/usr/bin/env python

#------------------------------------------------------------------------

import os
import sys
import shutil
import textwrap
from os.path import join

from distutils.core import Command, setup
from distutils.sysconfig import get_python_inc, get_config_var

from unittest import TextTestRunner
testrunner = TextTestRunner

#------------------------------------------------------------------------
# Top Level Packages
#------------------------------------------------------------------------

packages = [
    'blaze',
    'blaze.aterm',
    'blaze.carray',
    'blaze.datashape',
    'blaze.dist',
    'blaze.compile',
    'blaze.expr',
    'blaze.include',
    'blaze.layouts',
    'blaze.persistence',
    'blaze.rosetta',
    'blaze.rts',
    'blaze.sources',
]

#------------------------------------------------------------------------
# Minimum Versions
#------------------------------------------------------------------------

# The minimum version of Cython required for generating extensions
min_cython_version = '0.16'
# The minimum version of NumPy required
min_numpy_version = '1.5'
# The minimum version of llvmpy required
min_llvmpy_version = '0.8.4'
# The minimum version of Numba required
min_numba_version = '0.5'

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(
        initial_indent='   ', subsequent_indent='   ')

    print ".. %s:: %s" % (kind.upper(), head)
    for line in tw.wrap(body):
        print line

def exit_with_error(head, body=''):
    _print_admonition('error', head, body)
    sys.exit(1)

def print_warning(head, body=''):
    _print_admonition('warning', head, body)

def check_import(pkgname, pkgver):
    try:
        mod = __import__(pkgname)
    except ImportError:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run carray!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )
    else:
        if mod.__version__ < pkgver:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run carray!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )

    print ( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': pkgname, 'pkgver': mod.__version__} )
    globals()[pkgname] = mod

#------------------------------------------------------------------------
# Cython Sanity Check
#------------------------------------------------------------------------

try:
    from Cython.Distutils import Extension, build_ext
    from Cython.Compiler.Main import Version
except:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to compile carray!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version} )

if Version.version < min_cython_version:
    exit_with_error(
        "At least Cython %s is needed so as to generate extensions!"
        % (min_cython_version) )
else:
    print ( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': 'Cython', 'pkgver': Version.version} )

#------------------------------------------------------------------------
# Numpy Sanity Check
#------------------------------------------------------------------------

check_import('numpy', min_numpy_version)

#------------------------------------------------------------------------
# LLVM Sanity Check
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# C Compiler Environment
#------------------------------------------------------------------------

# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()

lib_dirs = []
libs = []
# Include NumPy header dirs
from numpy.distutils.misc_util import get_numpy_include_dirs
optional_libs = []

# Handle --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

# Add -msse2 flag for optimizing shuffle in include Blosc
if os.name == 'posix':
    CFLAGS.append("-msse2")

# Add some macros here for debugging purposes, if needed
def_macros = [('DEBUG', 0)]


#------------------------------------------------------------------------
# Extension
#------------------------------------------------------------------------

numpy_path = get_numpy_include_dirs()[0]

blosc_path  = 'blaze/include/blosc/'


carray_source = [
    "blaze/carray/carrayExtension.pyx"
]

carray_depends = [
]

blosc_source = [
    blosc_path + "blosc.c",
    blosc_path + "blosclz.c",
    blosc_path + "shuffle.c"
]

blosc_depends = [
    blosc_path + "blosc.h",
    blosc_path + "blosclz.h",
    blosc_path + "shuffle.h"
]

extensions = [
    Extension(
        "blaze.carray.carrayExtension",
        include_dirs=[
            blosc_path,
            numpy_path,
        ],

        sources = list(carray_source + blosc_source),
        depends = list(carray_depends + blosc_depends),

        library_dirs=lib_dirs,
        libraries=libs,
        extra_link_args=LFLAGS,
        extra_compile_args=CFLAGS,
        define_macros=def_macros,
   ),
   Extension(
        "blaze.ts.ucr_dtw.ucr",
        sources = ["blaze/ts/ucr_dtw/ucr.pyx", "blaze/ts/ucr_dtw/dtw.c"],
        depends = ["blaze/ts/ucr_dtw/dtw.h"],
        include_dirs = [numpy_path]
   ),
   Extension(
        "blaze.compile.executors", ["blaze/compile/executors.pyx"],
        include_dirs = [numpy_path],
   ),
   Extension(
       "blaze.sources.descriptors.lldescriptors",
       ["blaze/sources/descriptors/lldescriptors.pyx"],
       include_dirs = [],
   ),
   Extension(
       "blaze.sources.descriptors.llindexers",
       ["blaze/sources/descriptors/llindexers.pyx"],
       include_dirs = [],
   ),
   Extension(
        "blaze.cutils", ["blaze/cutils.pyx"],
        include_dirs = [numpy_path],
   ),
   Extension(
        "blaze.datashape.cdatashape", ["blaze/datashape/datashape.c"],
        include_dirs = [],
        define_macros=[('DEBUG', 1)]
   ),

   Extension(
        "blaze.algo.mean",
        sources = ["blaze/algo/mean.pyx"],
        include_dirs = [numpy_path]
   ),
]

#------------------------------------------------------------------------
# Commands
#------------------------------------------------------------------------

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = []

    def initialize_options(self):
        self._clean_me = []
        self._clean_trees = []

        for toplevel in packages:
            for root, dirs, files in list(os.walk(toplevel)):
                for f in files:
                    if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o', '.pyd'):
                        self._clean_me.append(join(root, f))

        for d in ('build',):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                print 'flushing', clean_me
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                print 'flushing', clean_tree
                shutil.rmtree(clean_tree)
            except Exception:
                pass

#------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------

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
    license='BSD',
    platforms = ['any'],
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
    packages=packages,
    install_requires=[''],
    ext_modules=extensions,
    cmdclass = {
        'build_ext' : build_ext,
        'test'      : testrunner,
        'clean'     : CleanCommand,
    },
)
