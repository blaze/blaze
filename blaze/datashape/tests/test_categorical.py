import unittest

import blaze
from blaze.datashape.coretypes import Enum

class TestCategorical(unittest.TestCase):

    def test_categorical_single(self):
        res = blaze.dshape('Categorical(Foo)')

        assert isinstance(res, Enum)

    def test_categorical_multi(self):
        res = blaze.dshape('Categorical(Foo, Bar)')

        assert isinstance(res, Enum)

    def test_categorical_module(self):
        source = """
        data Day = Monday | Tuesday | Wednesday | Thursday | Friday

        type Person = {
            days : Day;
        }
        """
        res = blaze.dshape(source, multi=True)

        assert isinstance(res[0], Enum)
