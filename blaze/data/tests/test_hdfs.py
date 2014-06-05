from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os
import csv

import datashape

from blaze.data.core import DataDescriptor
from blaze.data import CSV
from blaze.data import hdfs_open
from blaze.data import hdfs_isfile
from blaze.data.hdfs import DEFAULT_HOSTNAME, DEFAULT_PORT, host_port_path


class TestHDFS(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_host_port_path1(self):
        h, prt, path = host_port_path("hdfs://localhost:9000/user/local/testbuf.csv")
        self.assertEqual(h, "localhost")
        self.assertEqual(prt, 9000)
        self.assertEqual(path, "/user/local/testbuf.csv")

    def test_host_port_path2(self):
        h, prt, path = host_port_path("/user/local/testbuf.csv")
        self.assertEqual(h, DEFAULT_HOSTNAME)
        self.assertEqual(prt, DEFAULT_PORT)
        self.assertEqual(path, "/user/local/testbuf.csv")


if __name__ == '__main__':
    unittest.main()

