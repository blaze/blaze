from __future__ import absolute_import, division, print_function

import datashape

import blaze
import unittest
from blaze.catalog.tests.catalog_harness import CatalogHarness


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.cat = CatalogHarness()
        blaze.catalog.load_config(self.cat.catfile)

    def tearDown(self):
        blaze.catalog.load_default()
        self.cat.close()

    def test_dir_traversal(self):
        blaze.catalog.cd('/')
        self.assertEquals(blaze.catalog.cwd(), '/')
        self.assertEquals(blaze.catalog.ls(),
                          ['csv_arr', 'hdf5_arr', 'json_arr',
                           'npy_arr', 'py_arr', 'subdir'])
        self.assertEquals(blaze.catalog.ls_arrs(),
                          ['csv_arr', 'hdf5_arr', 'json_arr',
                           'npy_arr', 'py_arr'])
        self.assertEquals(blaze.catalog.ls_dirs(),
                          ['subdir'])
        blaze.catalog.cd('subdir')
        self.assertEquals(blaze.catalog.cwd(), '/subdir')
        self.assertEquals(blaze.catalog.ls(),
                          ['csv_arr2'])

    def test_load_csv(self):
        # Confirms that a simple csv file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('csv_arr')
        ds = datashape.dshape('5, {Letter: string; Number: int32}')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.dd_as_py(a._data)
        self.assertEqual(dat, [{'Letter': 'alpha', 'Number': 0},
                               {'Letter': 'beta', 'Number': 1},
                               {'Letter': 'gamma', 'Number': 2},
                               {'Letter': 'delta', 'Number': 3},
                               {'Letter': 'epsilon', 'Number': 4}])

    def test_load_json(self):
        # Confirms that a simple json file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('json_arr')
        ds = datashape.dshape('2, var, int32')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.dd_as_py(a._data)
        self.assertEqual(dat, [[1, 2, 3], [1, 2]])

    def test_load_npy(self):
        # Confirms that a simple npy file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('npy_arr')
        ds = datashape.dshape('20, {idx: int32; val: string}')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.dd_as_py(a._data)
        self.assertEqual([x['idx'] for x in dat],
                         list(range(20)))
        self.assertEqual([x['val'] for x in dat],
                         ['yes', 'no'] * 10)

    def test_load_py(self):
        # Confirms that a simple py file can generate a blaze array
        blaze.catalog.cd('/')
        a = blaze.catalog.get('py_arr')
        ds = datashape.dshape('5, int32')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.dd_as_py(a._data)
        self.assertEqual(dat, [1, 2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()
