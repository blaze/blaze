# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
from itertools import ifilter
from functools import partial
import subprocess

PY3 = sys.version_info[0] >= 3

#------------------------------------------------------------------------
# Test filtering
#------------------------------------------------------------------------

EXCLUDE_TEST_PACKAGES = [
    # "blaze.some_subpackage",
]

def make_path(root, predicate):
    "Call the predicate with a file path (e.g. blaze/test/foo.py)"
    return lambda item: predicate(os.path.join(root, item))

def qualify_path(root, predicate):
    "Call the predicate with a dotted name (e.g. blaze.tests.foo)"
    return make_path(root, lambda item: predicate(qualify_test_name(item)))


class Filter(object):
    def __init__(self, matcher=None):
        self.matcher = matcher

class PackageFilter(Filter):
    def filter(self, root, dirs, files):
        matcher = qualify_path(root, self.matcher)
        return ifilter(matcher, dirs), files

class ModuleFilter(Filter):
    def filter(self, root, dirs, files):
        matcher = qualify_path(root, self.matcher)
        return dirs, ifilter(matcher, files)

class FileFilter(Filter):
    def filter(self, root, dirs, files):
        return dirs, [fn for fn in files if fn.endswith(".py")]

#------------------------------------------------------------------------
# Test discovery
#------------------------------------------------------------------------

class Walker(object):
    def __init__(self, root, filters):
        self.root = root
        self.filters = filters

    def walk(self):
        for root, dirs, files in os.walk(self.root):
            dirs[:], files[:] = apply_filters(root, dirs, files, self.filters)
            yield ([os.path.join(root, dir) for dir in dirs],
                   [os.path.join(root, fn) for fn in files])


def apply_filters(root, dirs, files, filters):
    for filter in filters:
        dirs, files = filter.filter(root, dirs, files)

    return dirs, files

def qualify_test_name(root):
    root, ext = os.path.splitext(root)
    qname = root.replace("/", ".").replace("\\", ".").replace(os.sep, ".") + "."
    offset = qname.rindex('blaze.')
    return qname[offset:].rstrip(".")

def match(items, modname):
    return any(item in modname for item in items)

#------------------------------------------------------------------------
# Signal handling
#------------------------------------------------------------------------

def map_returncode_to_message(retcode):
    if retcode < 0:
        retcode = -retcode
        return signal_to_name.get(retcode, "Signal %d" % retcode)

    return ""

try:
    import signal
except ImportError:
    signal_to_name = {}
else:
    signal_to_name = dict((signal_code, signal_name)
                           for signal_name, signal_code in vars(signal).items()
                               if signal_name.startswith("SIG"))

#------------------------------------------------------------------------
# Test running
#------------------------------------------------------------------------

def test(whitelist=None, blacklist=None, print_failures_only=False):
    """
    Run tests under the blaze directory.
    """
    # Make some test filters
    filters = [
        PackageFilter(lambda pkg: not any(
            pkg.startswith(p) for p in EXCLUDE_TEST_PACKAGES)),
        PackageFilter(lambda pkg: not pkg.endswith(".__pycache__")),
        ModuleFilter(lambda modname: modname.split('.')[-1].startswith("test_")),
        FileFilter(),
    ]

    if whitelist:
        filters.append(ModuleFilter(partial(match, whitelist)))

    if blacklist:
        filters.append(ModuleFilter(lambda item: not match(blacklist, item)))

    # Run tests
    runner = TestRunner(print_failures_only)
    run_tests(runner, filters)

    sys.stdout.write("ran test files: failed: (%d/%d)\n" % (runner.failed,
                                                            runner.ran))

    return 0 if runner.failed == 0 else 1

def run_tests(test_runner, filters):
    """
    Run tests:

        - Find tests in packages called 'tests'
        - Run any test files under a 'tests' package or a subpackage
    """
    testpkg_walker = Walker("blaze", filters)

    for testpkgs, _ in testpkg_walker.walk():
        for testpkg in testpkgs:
            if os.path.basename(testpkg) == "tests":
                # print("testdir:", testpkg)
                test_walker = Walker(testpkg, filters)
                for _, testfiles in test_walker.walk():
                    for testfile in testfiles:
                        # print("testfile:", testfile)
                        modname = qualify_test_name(testfile)
                        test_runner.run(modname)


class TestRunner(object):
    """
    Test runner used by runtests.py
    """

    def __init__(self, print_failures_only):
        self.ran = 0
        self.failed = 0
        self.print_failures_only = print_failures_only

    def run(self, modname):
        self.ran += 1
        if not self.print_failures_only:
            sys.stdout.write("running %-61s" % (modname,))

        process = subprocess.Popen([sys.executable, '-m', modname],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        out, err = process.communicate()

        if process.returncode == 0:
            if not self.print_failures_only:
                sys.stdout.write("SUCCESS\n")
        else:
            if self.print_failures_only:
                sys.stdout.write("running %-61s" % (modname,))

            sys.stdout.write("FAILED: %s\n" % map_returncode_to_message(
                                            process.returncode))
            if PY3:
                out = str(out, encoding='UTF-8')
                err = str(err, encoding='UTF-8')
            sys.stdout.write(out)
            sys.stdout.write(err)
            sys.stdout.write("-" * 80)
            sys.stdout.write('\n')
            self.failed += 1
