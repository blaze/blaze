import unittest
import sys
import blaze
from blaze import datashape
from blaze.datadescriptor import (DyNDDataDescriptor,
                IDataDescriptor, IElementReader, IElementReadIter,
                dd_as_py)
from ...py3help import _inttypes
import ctypes

try:
    import dynd
    from dynd import nd, ndt
except ImportError:
    dynd = None

if sys.version_info >= (2, 7):
    from unittest import skipIf
else:
    from nose.plugins.skip import SkipTest
    class skipIf(object):
        def __init__(self, condition, reason):
            self.condition = condition
            self.reason = reason

        def __call__(self, func):
            if self.condition:
                from nose.plugins.skip import SkipTest
                def wrapped(*args, **kwargs):
                    raise SkipTest("Test %s is skipped because: %s" %
                                    (func.__name__, self.reason))
                wrapped.__name__ = func.__name__
                return wrapped
            else:
                return func

class TestDyNDDataDescriptor(unittest.TestCase):
    @skipIf(dynd is None, 'dynd is not installed')
    def test_basic_object_type(self):
        self.assertTrue(issubclass(DyNDDataDescriptor, IDataDescriptor))
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertEqual(dd_as_py(dd), [[1, 2, 3], [4, 5, 6]])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_descriptor_iter_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        # Iteration should produce DyNDDataDescriptor instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyNDDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [[1, 2, 3], [4, 5, 6]])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_descriptor_getitem_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        # Indexing should produce DyNDDataDescriptor instances
        self.assertTrue(isinstance(dd[0], DyNDDataDescriptor))
        self.assertEqual(dd_as_py(dd[0]), [1,2,3])
        self.assertTrue(isinstance(dd[1,2], DyNDDataDescriptor))
        self.assertEqual(dd_as_py(dd[1,2]), 6)

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_iter_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        # Requesting element iteration should produce an
        # IElementReadIter object
        ei = dd.element_read_iter()
        self.assertTrue(isinstance(ei, IElementReadIter))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        for ptr in ei:
            self.assertTrue(isinstance(ptr, _inttypes))

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_getitem_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        # Requesting get_element with one index should produce an
        # IElementReader object
        ge = dd.element_reader(1)
        self.assertTrue(isinstance(ge, IElementReader))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        self.assertTrue(isinstance(ge.read_single((1,)), _inttypes))

        # Requesting element reader with two indices should produce an
        # IElementReader object
        ge = dd.element_reader(2)
        self.assertTrue(isinstance(ge, IElementReader))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        self.assertTrue(isinstance(ge.read_single((1,2)), _inttypes))

    @skipIf(dynd is None, 'dynd is not installed')
    def test_var_dim(self):
        a = nd.ndobject([[1,2,3], [4,5], [6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd_as_py(dd), [[1,2,3], [4,5], [6]])
        self.assertEqual(dd_as_py(dd[0]), [1,2,3])
        self.assertEqual(dd_as_py(dd[1]), [4,5])
        self.assertEqual(dd_as_py(dd[2]), [6])

if __name__ == '__main__':
    unittest.main()
