# -*- coding: utf-8 -*-

from __future__ import absolute_import

import unittest
import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from blaze import blz
from blaze.blz.tests import common

class basicTest(common.MayBeDiskTest):

    def getobject(self):
        if self.flavor == 'barray':
            obj = blz.zeros(10, dtype="i1", rootdir=self.rootdir)
            self.assertEqual(type(obj), blz.barray)
        elif self.flavor == 'btable':
            obj = blz.fromiter(((i,i*2) for i in range(10)), dtype='i2,f4',
                              count=10, rootdir=self.rootdir)
            self.assertEqual(type(obj), blz.btable)
        return obj

    def test00a(self):
        """Creating attributes in a new barray."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        self.assertEqual(cn.attrs['attr1'], 'val1')
        self.assertEqual(cn.attrs['attr2'], 'val2')
        self.assertEqual(cn.attrs['attr3'], 'val3')
        self.assertEqual(len(cn.attrs), 3)

    def test00b(self):
        """Accessing attributes in a opened barray."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        # Re-open the barray
        if self.rootdir:
            cn = blz.open(rootdir=self.rootdir)
        self.assert_(cn.attrs['attr1'] == 'val1')
        self.assert_(cn.attrs['attr2'] == 'val2')
        self.assert_(cn.attrs['attr3'] == 'val3')
        self.assert_(len(cn.attrs) == 3)

    def test01a(self):
        """Removing attributes in a new barray."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        # Remove one of them
        del cn.attrs['attr2']
        self.assert_(cn.attrs['attr1'] == 'val1')
        self.assert_(cn.attrs['attr3'] == 'val3')
        self.assertRaises(KeyError, cn.attrs.__getitem__, 'attr2')
        self.assert_(len(cn.attrs) == 2)

    def test01b(self):
        """Removing attributes in a opened barray."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        # Reopen
        if self.rootdir:
            cn = blz.open(rootdir=self.rootdir)
        # Remove one of them
        del cn.attrs['attr2']
        self.assert_(cn.attrs['attr1'] == 'val1')
        self.assert_(cn.attrs['attr3'] == 'val3')
        self.assertRaises(KeyError, cn.attrs.__getitem__, 'attr2')
        self.assert_(len(cn.attrs) == 2)

    def test01c(self):
        """Appending attributes in a opened barray."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        # Reopen
        if self.rootdir:
            cn = blz.open(rootdir=self.rootdir)
        # Append attrs
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        self.assert_(cn.attrs['attr1'] == 'val1')
        self.assert_(cn.attrs['attr2'] == 'val2')
        self.assert_(cn.attrs['attr3'] == 'val3')
        self.assert_(len(cn.attrs) == 3)

    def test02(self):
        """Checking iterator in attrs accessor."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        count = 0
        for key, val in cn.attrs:
            if key == 'attr1':
                self.assert_(val, 'val1')
            if key == 'attr2':
                self.assert_(val, 'val2')
            if key == 'attr3':
                self.assert_(val, 'val3')
            count += 1
        self.assert_(count, 3)

class barrayTest(basicTest, unittest.TestCase):
    flavor = "barray"
    disk = False

class barrayDiskTest(basicTest, unittest.TestCase):
    flavor = "barray"
    disk = True

class btableTest(basicTest, unittest.TestCase):
    flavor = "btable"
    disk = False

class btableDiskTest(basicTest, unittest.TestCase):
    flavor = "btable"
    disk = True

if __name__ == '__main__':
    unittest.main(verbosity=2)

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
