import blaze
from blaze.datadescriptor import dd_as_py
import unittest


class TestBasic(unittest.TestCase):

    def test_add(self):
        # something is wrong with 'int8' --- ctype not converting correctly
        types = ['int16', 'int32', 'int64']
        for type_ in types:
            a = blaze.array(range(3), dshape=type_)
            c = ((a+a)*a).eval()
            self.assertEqual(dd_as_py(c._data), [0, 2, 8])

if __name__ == '__main__':
    unittest.main()