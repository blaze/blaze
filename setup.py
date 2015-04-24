#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os
from os.path import join, exists, isdir, abspath, dirname

import site
import shutil
import sys
from fnmatch import fnmatch

from distutils.core import setup

import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = os.path.join('blaze', '_version.py')
versioneer.versionfile_build = versioneer.versionfile_source
versioneer.tag_prefix = ''  # tags are like 1.2.0
versioneer.parentdir_prefix = 'blaze-'  # dirname like 'myproject-1.2.0'


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

def check_remove_blaze_install(site_packages):
    blaze_path = join(site_packages, "blaze")
    if not (exists(blaze_path) and isdir(blaze_path)):
        return
    prompt = "Found existing blaze install: %s\nRemove it? [y|N] " % blaze_path
    val = input(prompt)
    if val == "y":
        print("Removing old blaze install...", end=" ")
        try:
            shutil.rmtree(blaze_path)
            print("Done")
        except (IOError, OSError):
            print("Unable to remove old blaze at %s, exiting" % blaze_path)
            sys.exit(-1)
    else:
        print("Not removing old blaze install")
        sys.exit(1)


def find_data_files(exts, where='blaze'):
    exts = tuple(exts)
    for root, dirs, files in os.walk(where):
        for f in files:
            if any(fnmatch(f, pat) for pat in exts):
                yield os.path.join(root, f)

def getsitepackages():
    _is_64bit = (getattr(sys, 'maxsize', None)
                 or getattr(sys, 'maxint')) > 2**32
    _is_pypy = hasattr(sys, 'pypy_version_info')
    _is_jython = sys.platform[:4] == 'java'

    prefixes = [sys.prefix, sys.exec_prefix]

    sitepackages = []
    seen = set()
    for prefix in prefixes:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)

        if sys.platform in ('os2emx', 'riscos') or _is_jython:
            sitedirs = [os.path.join(prefix, "Lib", "site-packages")]
        elif _is_pypy:
            sitedirs = [os.path.join(prefix, 'site-packages')]
        elif sys.platform == 'darwin' and prefix == sys.prefix:
            if prefix.startswith("/System/Library/Frameworks/"):  # Apple's Python
                sitedirs = [os.path.join("/Library/Python", sys.version[:3], "site-packages"),
                            os.path.join(prefix, "Extras", "lib", "python")]

            else:  # any other Python distros on OSX work this way
                sitedirs = [os.path.join(prefix, "lib",
                            "python" + sys.version[:3], "site-packages")]

        elif os.sep == '/':
            sitedirs = [os.path.join(prefix,
                                     "lib",
                                     "python" + sys.version[:3],
                                     "site-packages"),
                        os.path.join(prefix, "lib", "site-python"),
                        ]
            lib64_dir = os.path.join(prefix, "lib64", "python" + sys.version[:3], "site-packages")
            if (os.path.exists(lib64_dir) and
                os.path.realpath(lib64_dir) not in [os.path.realpath(p) for p in sitedirs]):
                if _is_64bit:
                    sitedirs.insert(0, lib64_dir)
                else:
                    sitedirs.append(lib64_dir)
            try:
                # sys.getobjects only available in --with-pydebug build
                sys.getobjects
                sitedirs.insert(0, os.path.join(sitedirs[0], 'debug'))
            except AttributeError:
                pass
            # Debian-specific dist-packages directories:
            sitedirs.append(os.path.join(prefix, "local/lib",
                                         "python" + sys.version[:3],
                                         "dist-packages"))
            sitedirs.append(os.path.join(prefix, "lib",
                                         "python" + sys.version[:3],
                                         "dist-packages"))
            if sys.version_info[0] >= 3:
                sitedirs.append(os.path.join(prefix, "lib",
                                             "python" + sys.version[0],
                                             "dist-packages"))
            sitedirs.append(os.path.join(prefix, "lib", "dist-python"))
        else:
            sitedirs = [prefix, os.path.join(prefix, "lib", "site-packages")]
        if sys.platform == 'darwin':
            # for framework builds *only* we add the standard Apple
            # locations. Currently only per-user, but /Library and
            # /Network/Library could be added too
            if 'Python.framework' in prefix:
                home = os.environ.get('HOME')
                if home:
                    sitedirs.append(
                        os.path.join(home,
                                     'Library',
                                     'Python',
                                     sys.version[:3],
                                     'site-packages'))
        for sitedir in sitedirs:
            sitepackages.append(os.path.abspath(sitedir))

    sitepackages = [p for p in sitepackages if os.path.isdir(p)]
    return sitepackages



# Parse command line args
if 'develop' in sys.argv:
    if '--user' in sys.argv:
        site_packages = site.USER_SITE
    else:
        site_packages = getsitepackages()[0]

    check_remove_blaze_install(site_packages)
    path_file = join(site_packages, 'blaze.pth')
    path = abspath(dirname(__file__))
    with open((path_file), 'w+') as f:
        f.write(path)
    print("Install blaze for development:")
    print(" - writing path '%s' to %s" % (path, path_file))

packages = find_packages()
testdirs = find_packages(predicate=(lambda x: istestdir(x) and
                                    os.path.basename(x) == 'tests'))


exts = '*.h5', '*.csv', '*.xls', '*.xlsx', '*.db', '*.json', '*.gz', '*.hdf5'
package_data = [os.path.join(x.replace('blaze' + os.sep, ''), '*.py')
                for x in testdirs]
package_data += [x.replace('blaze' + os.sep, '')
                 for x in find_data_files(exts)]


with open('README.md') as f:
    longdesc = f.read()

with open('requirements-strict.txt') as f:
    install_requires = f.read().strip().split('\n')


setup(
    name='blaze',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='Blaze',
    long_description=longdesc,
    install_requires=install_requires,
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
    package_data={'blaze': package_data},
    packages=packages
)
