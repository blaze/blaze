import blaze
from blaze.datadescriptor import dd_as_py
from blaze.datashape import to_numpy, to_dtype
import numpy as np
import unittest
from .common import MayBeUriTest


class TestBasicTypes(unittest.TestCase):

    def test_ints(self):
        types = ['int8', 'int16', 'int32', 'int64']
        for type_ in types:
            a = blaze.array(np.arange(3), dshape=type_)
            dtype = to_dtype(a.dshape)
            self.assertEqual(dtype, np.dtype(type_))
            self.assertEqual(dd_as_py(a._data), [0, 1, 2])

    def test_uints(self):
        types = ['uint8', 'uint16', 'uint32', 'uint64']
        for type_ in types:
            a = blaze.array(np.arange(3), dshape=type_)
            dtype = to_dtype(a.dshape)
            self.assertEqual(dtype, np.dtype(type_))
            self.assertEqual(dd_as_py(a._data), [0, 1, 2])

    def test_floats(self):
        types = ['float16', 'float32', 'float64']
        for type_ in types:
            a = blaze.array(np.arange(3), dshape=type_)
            dtype = to_dtype(a.dshape)
            self.assertEqual(dtype, np.dtype(type_))
            if type_ != 'float16':
                # dd_as_py does not support this yet
                self.assertEqual(dd_as_py(a._data), [0, 1, 2])

    def test_complex(self):
        types = ['complex64', 'complex128']
        for type_ in types:
            a = blaze.array(np.arange(3), dshape=type_)
            dtype = to_dtype(a.dshape)
            self.assertEqual(dtype, np.dtype(type_))
            # dd_as_py does not support complexes yet..
            self.assertEqual(dd_as_py(a._data), [0, 1, 2])
