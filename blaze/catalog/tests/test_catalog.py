from __future__ import absolute_import, division, print_function

import unittest

import datashape
import blaze
from blaze.optional_packages import tables_is_here
from blaze.catalog.tests.catalog_harness import CatalogHarness
from blaze.py2help import skipIf


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
        entities = ['csv_arr', 'json_arr', 'npy_arr', 'py_arr', 'subdir']
        if tables_is_here:
            entities.append('hdf5_arr')
        self.assertEquals(blaze.catalog.ls(), sorted(entities))
        arrays = ['csv_arr', 'json_arr', 'npy_arr', 'py_arr']
        if tables_is_here:
            arrays.append('hdf5_arr')
        self.assertEquals(blaze.catalog.ls_arrs(), sorted(arrays))
        self.assertEquals(blaze.catalog.ls_dirs(),
                          ['hdf5_dir', 'subdir'])
        blaze.catalog.cd('subdir')
        self.assertEquals(blaze.catalog.cwd(), '/subdir')
        self.assertEquals(blaze.catalog.ls(),
                          ['csv_arr2'])

    def test_load_csv(self):
        # Confirms that a simple csv file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('csv_arr')
        ds = datashape.dshape('5 * {Letter: string, Number: int32}')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.ddesc_as_py(a.ddesc)
        self.assertEqual(dat, [{'Letter': 'alpha', 'Number': 0},
                               {'Letter': 'beta', 'Number': 1},
                               {'Letter': 'gamma', 'Number': 2},
                               {'Letter': 'delta', 'Number': 3},
                               {'Letter': 'epsilon', 'Number': 4}])

    def test_load_json(self):
        # Confirms that a simple json file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('json_arr')
        ds = datashape.dshape('2 * var * int32')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.ddesc_as_py(a.ddesc)
        self.assertEqual(dat, [[1, 2, 3], [1, 2]])

    @skipIf(not tables_is_here, 'PyTables is not installed')
    def test_load_hdf5(self):
        # Confirms that a simple hdf5 array in a file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('hdf5_arr')
        ds = datashape.dshape('2 * 3 * int32')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.ddesc_as_py(a.ddesc)
        self.assertEqual(dat, [[1, 2, 3], [3, 2, 1]])

    @skipIf(not tables_is_here, 'PyTables is not installed')
    def test_hdf5_dir(self):
        blaze.catalog.cd('/hdf5_dir')
        self.assertEquals(blaze.catalog.cwd(), '/hdf5_dir')
        self.assertEquals(blaze.catalog.ls(), sorted(['a1', 'mygroup']))
        self.assertEquals(blaze.catalog.ls_dirs(), sorted(['mygroup']))
        self.assertEquals(blaze.catalog.ls_arrs(), sorted(['a1']))

    @skipIf(not tables_is_here, 'PyTables is not installed')
    def test_hdf5_subdir(self):
        blaze.catalog.cd('/hdf5_dir/mygroup')
        self.assertEquals(blaze.catalog.cwd(), '/hdf5_dir/mygroup')
        self.assertEquals(blaze.catalog.ls(),
                          sorted(['a2', 'a3', 'mygroup2']))
        self.assertEquals(blaze.catalog.ls_dirs(), sorted(['mygroup2']))
        self.assertEquals(blaze.catalog.ls_arrs(), sorted(['a2', 'a3']))

    @skipIf(not tables_is_here, 'PyTables is not installed')
    def test_hdf5_subdir_get(self):
        blaze.catalog.cd('/hdf5_dir/mygroup')
        a = blaze.catalog.get('a3')
        ds = datashape.dshape('2 * 3 * int32')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.ddesc_as_py(a.ddesc)
        self.assertEqual(dat, [[1, 3, 2], [2, 1, 3]])

    @skipIf(not tables_is_here, 'PyTables is not installed')
    def test_hdf5_subdir_ls(self):
        # Check top level
        blaze.catalog.cd('/')
        lall = blaze.catalog.ls_dirs()
        self.assertEqual(lall, ['hdf5_dir', 'subdir'])
        # Check HDF5 root level
        blaze.catalog.cd('/hdf5_dir')
        larrs = blaze.catalog.ls_arrs()
        self.assertEqual(larrs, ['a1'])
        ldirs = blaze.catalog.ls_dirs()
        self.assertEqual(ldirs, ['mygroup'])
        lall = blaze.catalog.ls()
        self.assertEqual(lall, ['a1', 'mygroup'])
        # Check HDF5 second level
        blaze.catalog.cd('/hdf5_dir/mygroup')
        larrs = blaze.catalog.ls_arrs()
        self.assertEqual(larrs, ['a2', 'a3'])
        ldirs = blaze.catalog.ls_dirs()
        self.assertEqual(ldirs, ['mygroup2'])
        lall = blaze.catalog.ls()
        self.assertEqual(lall, ['a2', 'a3', 'mygroup2'])

    def test_load_npy(self):
        # Confirms that a simple npy file can be loaded
        blaze.catalog.cd('/')
        a = blaze.catalog.get('npy_arr')
        ds = datashape.dshape('20 * {idx: int32, val: string}')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.ddesc_as_py(a.ddesc)
        self.assertEqual([x['idx'] for x in dat],
                         list(range(20)))
        self.assertEqual([x['val'] for x in dat],
                         ['yes', 'no'] * 10)

    def test_load_py(self):
        # Confirms that a simple py file can generate a blaze array
        blaze.catalog.cd('/')
        a = blaze.catalog.get('py_arr')
        ds = datashape.dshape('5 * int32')
        self.assertEqual(a.dshape, ds)
        dat = blaze.datadescriptor.ddesc_as_py(a.ddesc)
        self.assertEqual(dat, [1, 2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()
