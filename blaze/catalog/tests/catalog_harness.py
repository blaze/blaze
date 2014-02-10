"""
Some functions to create/tear down a simple catalog
for tests to use.
"""
from __future__ import absolute_import, division, print_function

import blaze
from blaze.optional_packages import tables_is_here
import numpy as np
from dynd import nd, ndt
import tempfile
import os
import shutil


class CatalogHarness(object):
    def __init__(self):
        self.catdir = tempfile.mkdtemp()
        self.arrdir = os.path.join(self.catdir, 'arrays')
        os.mkdir(self.arrdir)
        self.catfile = os.path.join(self.catdir, 'testcat.yaml')
        with open(self.catfile, 'w') as f:
            f.write('# Temporary catalog for Blaze testing\n')
            f.write('root: ./arrays\n')
        # Create arrays with various formats at the top level
        self.create_csv('csv_arr')
        if tables_is_here:
            self.create_hdf5('hdf5')
        self.create_npy('npy_arr')
        self.create_py('py_arr')
        self.create_json('json_arr')
        # Create an array in a subdirectory
        os.mkdir(os.path.join(self.arrdir, 'subdir'))
        self.create_csv('subdir/csv_arr2')

    def close(self):
        shutil.rmtree(self.catdir)

    def create_csv(self, name):
        with open(os.path.join(self.arrdir, '%s.csv' % name), 'w') as f:
            f.write('Letter, Number\n')
            f.write('alpha, 0\n')
            f.write('beta, 1\n')
            f.write('gamma, 2\n')
            f.write('delta, 3\n')
            f.write('epsilon, 4\n')
        with open(os.path.join(self.arrdir, '%s.array' % name), 'w') as f:
            f.write('type: csv\n')
            f.write('import: {\n')
            f.write('    headers: True\n')
            f.write('}\n')
            f.write('datashape: |\n')
            f.write('    var, {\n')
            f.write('        Letter: string;\n')
            f.write('        Number: int32;\n')
            f.write('    }\n')

    def create_json(self, name):
        a = nd.array([[1, 2, 3], [1, 2]])
        with open(os.path.join(self.arrdir, '%s.json' % name), 'w') as f:
            f.write(nd.as_py(nd.format_json(a)))
        with open(os.path.join(self.arrdir, '%s.array' % name), 'w') as f:
            f.write('type: json\n')
            f.write('import: {}\n')
            f.write('datashape: "var, var, int32"\n')

    def create_hdf5(self, name):
        import tables as tb
        a1 = nd.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        a2 = nd.array([[1, 2, 3], [3, 2, 1]], dtype="int32")
        fname = os.path.join(self.arrdir, '%s_arr.h5' % name)
        with tb.open_file(fname, 'w') as f:
            f.create_array(f.root, "a1", nd.as_numpy(a1))
            mg = f.create_group(f.root, "mygroup")
            f.create_array(mg, "a2", nd.as_numpy(a2))
        # Create a .array file for locating the dataset inside the file
        with open(os.path.join(self.arrdir, '%s_arr.array' % name), 'w') as f:
            f.write('type: hdf5\n')
            f.write('import: {\n')
            f.write('    datapath: /mygroup/a2\n')
            f.write('    }\n')
        # Create a .dir file for listing datasets inside the file
        with open(os.path.join(self.arrdir, '%s_dir.dir' % name), 'w') as f:
            f.write('type: hdf5\n')
            f.write('import: {\n')
            f.write('    filename: %s/\n' % fname)
            f.write('    }\n')

    def create_npy(self, name):
        a = np.empty(20, dtype=[('idx', np.int32), ('val', 'S4')])
        a['idx'] = np.arange(20)
        a['val'] = ['yes', 'no'] * 10
        np.save(os.path.join(self.arrdir, '%s.npy' % name), a)
        with open(os.path.join(self.arrdir, '%s.array' % name), 'w') as f:
            f.write('type: npy\n')
            f.write('import: {}\n')
            f.write('datashape: |\n')
            f.write('    M, {\n')
            f.write('        idx: int32;\n')
            f.write('        val: string;\n')
            f.write('    }\n')

    def create_py(self, name):
        with open(os.path.join(self.arrdir, '%s.py' % name), 'w') as f:
            f.write('import blaze\n')
            f.write('result = blaze.array([1, 2, 3, 4, 5])\n')
        with open(os.path.join(self.arrdir, '%s.array' % name), 'w') as f:
            f.write('type: py\n')
            f.write('import: {}\n')
            f.write('datashape: "5, int32"\n')
