#!/usr/bin/env python

#------------------------------------------------------------------------

from __future__ import absolute_import

import os
import sys
import shutil
import textwrap
from os.path import join

from distutils.core import Command, setup
from distutils.sysconfig import get_python_inc, get_config_var
from distutils.command.build import build
from distutils.command.build_ext import build_ext

#------------------------------------------------------------------------
# Top Level Packages
#------------------------------------------------------------------------

packages = [
    'blaze',
    'blaze.aterm',
    'blaze.aterm.tests',
    'blaze.blir',
    'blaze.blz',
    'blaze.blz.tests',
    'blaze.cgen',
    'blaze.ckernel',
    'blaze.ckernel.tests',
    'blaze.datadescriptor',
    'blaze.datadescriptor.tests',
    'blaze.datashape',
    'blaze.datashape.tests',
    'blaze.executive',
    'blaze.tests',
    'blaze._printing',
]

#------------------------------------------------------------------------
# Minimum Versions
#------------------------------------------------------------------------

min_cython_version = '0.16'
min_numpy_version  = '1.5'
min_llvmpy_version = '0.11.1'

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(
        initial_indent='   ', subsequent_indent='   ')

    print(".. %s:: %s" % (kind.upper(), head))
    for line in tw.wrap(body):
        print(line)

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
                "You need %(pkgname)s %(pkgver)s or greater to run Blaze!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )
    else:
        if mod.__version__ < pkgver:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run Blaze!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )

    print("* Found %(pkgname)s %(pkgver)s package installed."
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
        "You need %(pkgname)s %(pkgver)s or greater to compile Blaze!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version} )

if Version.version < min_cython_version:
    exit_with_error(
        "At least Cython %s is needed so as to generate extensions!"
        % (min_cython_version) )
else:
    print( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': 'Cython', 'pkgver': Version.version} )

#------------------------------------------------------------------------
# Numpy Sanity Check
#------------------------------------------------------------------------

check_import('numpy', min_numpy_version)

#------------------------------------------------------------------------
# LLVM Sanity Check
#------------------------------------------------------------------------

#check_import('llvmpy', min_llvmpy_version)

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

blosc_path  = 'blaze/blz/include/blosc/'


blz_source = [
    "blaze/blz/blz_ext.pyx"
]

blz_depends = [
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
        "blaze.blz.blz_ext",
        include_dirs=[
            blosc_path,
            numpy_path,
        ],

        sources = list(blz_source + blosc_source),
        depends = list(blz_depends + blosc_depends),

        library_dirs=lib_dirs,
        libraries=libs,
        extra_link_args=LFLAGS,
        extra_compile_args=CFLAGS,
        define_macros=def_macros,
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
                print('flushing', clean_me)
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                print('flushing', clean_tree)
                shutil.rmtree(clean_tree)
            except Exception:
                pass

#------------------------------------------------------------------------
# Parser
#------------------------------------------------------------------------

# Clang and libclang.so is hard dependency of Blaze at the
# moment. On Linux this is included with Anaconda, on other
# platforms it's up to the user to install clang with XCode.

# Blir requires the LLVM math intrinsics which need to be in the
# process space with libm. This will be fixed when we have
# support OpenLibM that works across all plaforms.

def on_path(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return False

class BuildParser(build):
    """ Build the parse tables for datashape """

    def run(self):
        # run the default build command first
        build.run(self)
        print('Rebuilding the datashape parser...')
        # add the build destination to the module path so we can load it
        sys.path.insert(0, self.build_lib)
        from blaze.datashape.parser import rebuild
        rebuild()

class BuildPrelude(build_ext):
    """ Build the Blir prelude """

    def run(self):
        if sys.platform == 'win32':
            print('WARNING: Skipping build of prelude on Windows')
        else:
            assert on_path('clang') or on_path('gcc') or on_path('minggw')
            extensions.append(
                Extension(
                     "blaze.blir.prelude",
                     sources = ["blaze/blir/prelude.c"],
                     depends = [],
                     include_dirs = [],
                )
            )
        build_ext.run(self)

#------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------

try:
    assert sys.version[0] != 3
except AssertionError:
    print("Python 3 is not supported as this time.")

# Kernel tree code requires clang++ for LLVM generation.

try:
    assert on_path('clang++')
except AssertionError:
    print("Clang++ is required to build Blaze.")

longdesc = open('README.md').read()

setup(
    name='blaze',
    version='0.1',
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='Blaze',
    long_description=longdesc,
    data_files=[],
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
    ext_modules=extensions,
    cmdclass = {
        'build_ext' : build_ext,
        'clean'     : CleanCommand,
        'build'     : BuildParser,
        'build_ext' : BuildPrelude,
    },
    scripts=['bin/blirc'],
)
