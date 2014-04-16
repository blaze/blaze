import unittest

from dynd import nd

from blaze.user import validate
from blaze import array


class Test_Validate(unittest.TestCase):
    def test_blaze(self):
        assert validate('2 * int', array([1, 2]))
        assert not validate('3 * int', array([1, 2]))

    def test_dynd(self):
        assert validate('2 * int', nd.array([1, 2], dtype='2 * int32'))
        assert not validate('3 * int', nd.array([1, 2], dtype='2 * int32'))
