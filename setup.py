#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import os
import sys
import shutil
import textwrap
from fnmatch import fnmatch

from setuptools import Command, setup, convert_path


#------------------------------------------------------------------------
# Top Level Packages
#------------------------------------------------------------------------

def ispackage(x):
    return os.path.isdir(x) and os.path.exists(os.path.join(x, '__init__.py'))


def istestdir(x):
    return os.path.isdir(x) and not os.path.exists(os.path.join(x, '__init__.py'))


def find_packages(where='blaze', exclude=('ez_setup', 'distribute_setup'),
                  predicate=ispackage):
    if sys.version_info[0] == 3:
        exclude += ('*py2only*', '*__pycache__*')

    func = lambda x: predicate(x) and not any(fnmatch(x, exc)
                                              for exc in exclude)
    return list(filter(func, [x[0] for x in os.walk(convert_path(where))]))


packages = find_packages()
testdirs = find_packages(predicate=(lambda x: istestdir(x) and
                                    os.path.basename(x) == 'tests'))

#------------------------------------------------------------------------
# Minimum Versions
#------------------------------------------------------------------------

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

def find_data_files(exts, where='blaze'):
    exts = tuple(exts)
    for root, dirs, files in os.walk(where):
        for f in files:
            if any(fnmatch(f, pat) for pat in exts):
                yield os.path.join(root, f)


exts = '*.h5', '*.csv', '*.xls', '*.xlsx', '*.db', '*.json', '*.gz', '*.hdf5'
package_data = [os.path.join(x.replace('blaze' + os.sep, ''),
                             '*.py') for x in testdirs]
package_data += [x.replace('blaze' + os.sep, '') for x in find_data_files(exts)]


with open('README.md') as f:
    longdesc = f.read()


setup(
    name='blaze',
    version='0.7.2',
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='Blaze',
    long_description=longdesc,
    install_requires=open('requirements-strict.txt').read().strip().split('\n'),
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
    package_data={'blaze': package_data},
    packages=packages,
    cmdclass={'clean': CleanCommand}
)
