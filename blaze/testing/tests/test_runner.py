# -*- coding: utf-8 -*-

"""
Test the test runner itself.
"""

from __future__ import print_function, division, absolute_import

import unittest
from blaze.testing import main, skip

def test_run():
    assert 5 == 5, "right?"

@skip("skip..")
def test_something():
    assert 5 == 4, "what?"

@skip("skip me!")
def test_skipping():
    assert not "Was I not skipped?"

class Test(unittest.TestCase):

    @skip("skip me!")
    def test_blah(self):
        assert 3 == 2

    def test_blah2(self):
        assert 3 == 3

if __name__ == '__main__':
    testprogram = main(exit=False)
    result = testprogram.result

    assert result.testsRun == 4
    assert len(result.failures) == 2
    assert len(result.skipped) == 1

    # See whether the test runner ran as main
    # raise Exception("ran!")