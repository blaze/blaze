import unittest

import blaze
from blaze.datashape.coretypes import Option

class TestOption(unittest.TestCase):

    def test_categorical_single(self):
        res = blaze.dshape('Option(int32)')

        assert isinstance(res, Option)

    def test_categorical_multi(self):
        res = blaze.dshape('2, 3, Option(int32)')

        assert isinstance(res[2], Option)
