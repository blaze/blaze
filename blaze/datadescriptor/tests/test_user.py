import unittest

from dynd import nd

from blaze.datadescriptor.user import like
from blaze.usability import like
from blaze import array

class Test_like(unittest.TestCase):
    def test_DDesc(self):
        from blaze.datadescriptor import DyND_DDesc
        dd = DyND_DDesc(nd.array([1, 2]))
        self.assertEqual(like([], dd),
                         [1, 2])
        self.assertEqual(nd.as_py(like(nd.array(), dd)),
                         nd.as_py(nd.array([1, 2])))
        b = like(array(0), dd)
        self.assertEqual(like([], b), like([], dd))
