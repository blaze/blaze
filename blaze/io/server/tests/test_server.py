from __future__ import absolute_import, division, print_function

import os
import sys
import random
import subprocess
import socket
import time
import blaze
import unittest
from blaze.catalog.tests.catalog_harness import CatalogHarness

from blaze.datadescriptor import dd_as_py
from blaze.io.client.rarray import RArray


class TestServer(unittest.TestCase):
    def startServer(self):
        # Start the server
        serverpy = os.path.join(os.path.dirname(__file__),
                                'start_simple_server.py')
        for attempt in range(5):
            self.port = 10000 + random.randrange(30000)
            cflags = 0
            if sys.platform == 'win32':
                cflags |= subprocess.CREATE_NEW_PROCESS_GROUP

            self.proc = subprocess.Popen([sys.executable,
                                          serverpy,
                                          self.cat.catfile,
                                          str(self.port)],
                                         executable=sys.executable,
                                         creationflags=cflags)
            for i in range(10):
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
        ra = RArray('%s/csv_arr' % self.baseurl)
        la = blaze.catalog.get('/csv_arr')
        self.assertEqual(la.dshape, ra.dshape)
        self.assertEqual(dd_as_py(la._data), dd_as_py(ra.get_data()._data))

if __name__ == '__main__':
    unittest.main()
