# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import inspect
import logging
import unittest
import functools

def make_unit_tests(global_scope):
    """
    Put FunctionTestCases in the global scope
    """
    module_name = global_scope['__name__']

    class BlazeTestCollection(unittest.TestCase):
        pass

    global_scope["BlazeTestCollection"] = BlazeTestCollection

    for name, obj in global_scope.items():
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            def wrapper(self, func=obj):
                return func()

            setattr(BlazeTestCollection, name, wrapper)

def main(**kwargs):
    if '-d' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        sys.argv.remove('-d')

    if '-D' in sys.argv:
        logging.getLogger().setLevel(logging.NOTSET)
        sys.argv.remove('-D')

    make_unit_tests(sys._getframe(1).f_globals)

    return unittest.main(**kwargs)

def parametrize(*parameters):
    """
    @parametrize('foo', 'bar')
    def test_func(foo_or_bar):
        print foo_or_bar # prints 'foo' or 'bar'

    Generates a unittest TestCase in the function's global scope named
    'test_func_testcase' with parametrized test methods.

    ':return: The original function
    """
    def decorator(func):
        class TestCase(unittest.TestCase):
            pass

        TestCase.__name__ = func.__name__

        for i, parameter in enumerate(parameters):
            name = '%s_%d' % (func.__name__, i)

            def testfunc(self, parameter=parameter):
                return func(parameter)

            testfunc.__name__ = name
            if func.__doc__:
                testfunc.__doc__ = func.__doc__.replace(func.__name__, name)

            # func.func_globals[name] = unittest.FunctionTestCase(testfunc)
            setattr(TestCase, name, testfunc)


        func.__globals__[func.__name__ + '_testcase'] = TestCase
        return func

    return decorator

#------------------------------------------------------------------------
# Support for unittest in < py2.7
#------------------------------------------------------------------------

have_unit_skip = sys.version_info[:2] > (2, 6)

if have_unit_skip:
    from unittest import SkipTest
else:
    class SkipTest(Exception):
        "Skip a test in < py27"

def skip_test(reason):
    if have_unit_skip:
        raise SkipTest(reason)
    else:
        print("Skipping: " + reason, file=sys.stderr)

def skip_if(should_skip, message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if should_skip:
                skip_test(message)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def skip_unless(should_skip, message):
    return skip_if(not should_skip, message)

def skip(message):
    return skip_if(True, message)

def checkSkipFlag(reason):
    def _checkSkipFlag(fn):
        def _checkSkipWrapper(self, *args, **kws):
            skip_test(reason)
        return _checkSkipWrapper
    return _checkSkipFlag