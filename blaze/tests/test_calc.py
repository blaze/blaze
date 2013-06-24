import blaze
from blaze.datadescriptor import dd_as_py
import unittest


class TestBasic(unittest.TestCase):

    def test_add(self):
        types = ['int8', 'int16', 'int32', 'int64']
        for type_ in types:
            a = blaze.array(range(3), dshape=type_)
            c = blaze.eval(((a+a)*a))
            self.assertEqual(dd_as_py(c._data), [0, 2, 8])

    #FIXME:  Need to convert uint8 from dshape to ctypes
    #        in _get_ctypes of blaze_kernel.py
    def test_mixed(self):
	types1 = ['int8', 'int16', 'int32', 'int64']
        types2 = ['int16', 'int32', 'float32', 'float64']
 	for ty1, ty2 in zip(types1, types2):
            a = blaze.array(range(1,6), dshape=ty1)
            b = blaze.array(range(5), dshape=ty2)
            c = (a+b)*(a-b) 
            c = blaze.eval(c)
            result = [a*a - b*b for (a,b) in zip(range(1,6),range(5))]
            self.assertEqual(dd_as_py(c._data), result)
if __name__ == '__main__':
    unittest.main()
