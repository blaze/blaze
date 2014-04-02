from __future__ import absolute_import, division, print_function

import unittest

import datashape
from dynd import nd
from blaze.datadescriptor import DyND_DDesc, DDesc, ddesc_as_py


class TestDyND_DDesc(unittest.TestCase):
    def test_basic_object_type(self):
        self.assertTrue(issubclass(DyND_DDesc, DDesc))
        a = nd.array([[1, 2, 3], [4, 5, 6]])
        dd = DyND_DDesc(a)
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, DDesc))
        self.assertEqual(ddesc_as_py(dd), [[1, 2, 3], [4, 5, 6]])

    def test_descriptor_iter_types(self):
        a = nd.array([[1, 2, 3], [4, 5, 6]])
        dd = DyND_DDesc(a)

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int32'))
        # Iteration should produce DyND_DDesc instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyND_DDesc))
            self.assertTrue(isinstance(el, DDesc))
            vals.append(ddesc_as_py(el))
        self.assertEqual(vals, [[1, 2, 3], [4, 5, 6]])

    def test_descriptor_getitem_types(self):
        a = nd.array([[1, 2, 3], [4, 5, 6]])
        dd = DyND_DDesc(a)

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int32'))
        # Indexing should produce DyND_DDesc instances
        self.assertTrue(isinstance(dd[0], DyND_DDesc))
        self.assertEqual(ddesc_as_py(dd[0]), [1,2,3])
        self.assertTrue(isinstance(dd[1,2], DyND_DDesc))
        self.assertEqual(ddesc_as_py(dd[1,2]), 6)

    def test_var_dim(self):
        a = nd.array([[1, 2, 3], [4, 5], [6]])
        dd = DyND_DDesc(a)

        self.assertEqual(dd.dshape, datashape.dshape('3 * var * int32'))
        self.assertEqual(ddesc_as_py(dd), [[1, 2, 3], [4, 5], [6]])
        self.assertEqual(ddesc_as_py(dd[0]), [1, 2, 3])
        self.assertEqual(ddesc_as_py(dd[1]), [4, 5])
        self.assertEqual(ddesc_as_py(dd[2]), [6])


if __name__ == '__main__':
    unittest.main()
