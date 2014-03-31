from __future__ import absolute_import, division, print_function

import os
import sys
import random
import subprocess
import socket
import time
import unittest

import datashape
import blaze
from blaze.catalog.tests.catalog_harness import CatalogHarness
from blaze.datadescriptor import ddesc_as_py, RemoteDataDescriptor


class TestServer(unittest.TestCase):
    def startServer(self):
        # Start the server
        serverpy = os.path.join(os.path.dirname(__file__),
                                'start_simple_server.py')
        for attempt in range(2):
            self.port = 10000 + random.randrange(30000)
            cflags = 0
            exe = sys.executable
            if sys.platform == 'win32':
                if sys.version_info[:2] > (2, 6):
                    cflags |= subprocess.CREATE_NEW_PROCESS_GROUP

            self.proc = subprocess.Popen([sys.executable,
                                          serverpy,
                                          self.cat.catfile,
                                          str(self.port)],
                                         executable=exe,
                                         creationflags=cflags)
            for i in range(30):
                time.sleep(0.2)
                if self.proc.poll() is not None:
                    break
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if s.connect_ex(('127.0.0.1',self.port)) == 0:
                    s.close()
                    return
                s.close()
            print("Couldn't start Blaze test server attempt %d" % attempt)
            self.proc.terminate()
        raise RuntimeError('Failed to start the test Blaze server')

    def setUp(self):
        self.cat = CatalogHarness()
        # Load the test catalog for comparison with the server
        blaze.catalog.load_config(self.cat.catfile)
        self.startServer()
        self.baseurl = 'http://localhost:%d' % self.port

    def tearDown(self):
        self.proc.terminate()
        blaze.catalog.load_default()
        self.cat.close()

    def test_get_arr(self):
        ra = blaze.array(RemoteDataDescriptor('%s/csv_arr' % self.baseurl))
        la = blaze.catalog.get('/csv_arr')
        self.assertEqual(la.dshape, ra.dshape)
        self.assertEqual(ddesc_as_py(la._data), ddesc_as_py(blaze.eval(ra)._data))

    def test_compute(self):
        ra = blaze.array(RemoteDataDescriptor('%s/py_arr' % self.baseurl))
        result = ra + 1
        result = blaze.eval(result)
        self.assertEqual(result.dshape, datashape.dshape('5 * int32'))
        self.assertEqual(ddesc_as_py(result._data), [2, 3, 4, 5, 6])

if __name__ == '__main__':
    unittest.main()
