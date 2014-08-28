#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import os
import sys
import shutil
import textwrap
from fnmatch import fnmatchcase

from distutils.core import Command, setup
from distutils.util import convert_path


#------------------------------------------------------------------------
# Top Level Packages
#------------------------------------------------------------------------

def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))

    if sys.version_info[0] == 3:
        exclude = exclude + ('*py2only*', )

    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]

    return out

packages = find_packages()

#------------------------------------------------------------------------
# Minimum Versions
#------------------------------------------------------------------------

min_numpy_version  = '1.5'

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
# Numpy Sanity Check
#------------------------------------------------------------------------

check_import('numpy', min_numpy_version)

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
                        self._clean_me.append(os.path.join(root, f))

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
# Setup
#------------------------------------------------------------------------

longdesc = open('README.md').read()

setup(
    name='blaze',
    version='0.6.2',
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
    cmdclass = {
        'clean'     : CleanCommand,
    }
)
