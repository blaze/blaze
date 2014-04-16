import unittest

from dynd import nd

from blaze.datadescriptor.user import into
from blaze import array

class Test_Into(unittest.TestCase):
    def test_DDesc(self):
        from blaze.datadescriptor import DyND_DDesc
        dd = DyND_DDesc(nd.array([1, 2]))
        self.assertEqual(into([], dd),
                         [1, 2])
        self.assertEqual(nd.as_py(into(nd.array(), dd)),
                         nd.as_py(nd.array([1, 2])))
        b = into(array(0), dd)
        self.assertEqual(into([], b), into([], dd))
