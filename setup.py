#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import os
import sys
from fnmatch import fnmatch

from setuptools import setup

import versioneer


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
    return list(filter(func, [x[0] for x in os.walk(where)]))


packages = find_packages()
testdirs = find_packages(predicate=(lambda x: istestdir(x) and
                                    os.path.basename(x) == 'tests'))


def find_data_files(exts, where='blaze'):
    exts = tuple(exts)
    for root, dirs, files in os.walk(where):
        for f in files:
            if any(fnmatch(f, pat) for pat in exts):
                yield os.path.join(root, f)


exts = '*.h5', '*.tsv', '*.csv', '*.xls', '*.xlsx', '*.db', '*.json', '*.gz', '*.hdf5'
package_data = [os.path.join(x.replace('blaze' + os.sep, ''), '*.py')
                for x in testdirs]
package_data += [x.replace('blaze' + os.sep, '')
                 for x in find_data_files(exts)]


with open('README.rst') as f:
    longdesc = f.read()

with open('requirements-strict.txt') as f:
    install_requires = f.read().strip().splitlines()


if sys.version_info[0] == 2:
    with open('requirements-py2.txt') as f:
        install_requires.extend(f.read().strip().splitlines())


if __name__ == '__main__':
    setup(name='blaze',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          author='Continuum Analytics',
          author_email='blaze-dev@continuum.io',
          description='Blaze',
          long_description=longdesc,
          install_requires=install_requires,
          license='BSD',
          classifiers=['Development Status :: 2 - Pre-Alpha',
                       'Environment :: Console',
                       'Intended Audience :: Developers',
                       'Intended Audience :: Science/Research',
                       'Intended Audience :: Education',
                       'License :: OSI Approved :: BSD License',
                       'Operating System :: OS Independent',
                       'Programming Language :: Python',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Utilities'],
          entry_points={'console_scripts': ['blaze-server = blaze.server.spider:_main']},
          package_data={'blaze': package_data},
          packages=packages)
