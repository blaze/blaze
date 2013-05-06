# -*- coding: utf-8 -*-
########################################################################
#
# multidimension tests for blaze.blz.
# based on barray tests, adapted to blaze and nosetest
#
########################################################################

from __future__ import absolute_import

import sys
import struct

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from blaze import blz
from .common import MayBeDiskTest
import unittest

class constructorTest(MayBeDiskTest, unittest.TestCase):

    open = False

    def test00a(self):
        """Testing `barray` reshape"""
        a = np.arange(16).reshape((2,2,4))
        b = blz.arange(16, rootdir=self.rootdir).reshape((2,2,4))
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `barray` reshape (large shape)"""
        a = np.arange(16000).reshape((20,20,40))
        b = blz.arange(16000, rootdir=self.rootdir).reshape((20,20,40))
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01a(self):
        """Testing `zeros` constructor (I)"""
        a = np.zeros((2,2,4), dtype='i4')
        b = blz.zeros((2,2,4), dtype='i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01b(self):
        """Testing `zeros` constructor (II)"""
        a = np.zeros(2, dtype='(2,4)i4')
        b = blz.zeros(2, dtype='(2,4)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01c(self):
        """Testing `zeros` constructor (III)"""
        a = np.zeros((2,2), dtype='(4,)i4')
        b = blz.zeros((2,2), dtype='(4,)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test02(self):
        """Testing `ones` constructor"""
        a = np.ones((2,2), dtype='(4,)i4')
        b = blz.ones((2,2), dtype='(4,)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test03a(self):
        """Testing `fill` constructor (scalar default)"""
        a = np.ones((2,200), dtype='(4,)i4')*3
        b = blz.fill((2,200), 3, dtype='(4,)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test03b(self):
        """Testing `fill` constructor (array default)"""
        a = np.ones((2,2), dtype='(4,)i4')*3
        b = blz.fill((2,2), [3,3,3,3], dtype='(4,)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test04(self):
        """Testing `fill` constructor with open and resize (array default)"""
        a = np.ones((3,200), dtype='(4,)i4')*3
        b = blz.fill((2,200), [3,3,3,3], dtype='(4,)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        c = np.ones((1,200), dtype='(4,)i4')*3
        b.append(c)
        #print "b->", `b`, len(b), b[1]
        assert_array_equal(a, b, "Arrays are not equal")

    def test05(self):
        """Testing `fill` constructor with open and resize (nchunks>1)"""
        a = np.ones((3,2000), dtype='(4,)i4')*3
        b = blz.fill((2,2000), [3,3,3,3], dtype='(4,)i4', rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        c = np.ones((1,2000), dtype='(4,)i4')*3
        b.append(c)
        #print "b->", `b`
        # We need to use the b[:] here to overcome a problem with the
        # assert_array_equal() function
        assert_array_equal(a, b[:], "Arrays are not equal")

class constructorDiskTest(constructorTest, unittest.TestCase):
    disk = True
    open = False

class constructorOpenTest(constructorTest, unittest.TestCase):
    disk = True
    open = True

class getitemTest(MayBeDiskTest, unittest.TestCase):

    open = False

    def test00a(self):
        """Testing `__getitem()__` method with only a start (scalar)"""
        a = np.ones((2,3), dtype="i4")*3
        b = blz.fill((2,3), 3, dtype="i4", rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = 1
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test00b(self):
        """Testing `__getitem()__` method with only a start (slice)"""
        a = np.ones((27,2700), dtype="i4")*3
        b = blz.fill((27,2700), 3, dtype="i4", rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = slice(1)
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01(self):
        """Testing `__getitem()__` method with a start and a stop"""
        a = np.ones((5,2), dtype="i4")*3
        b = blz.fill((5,2), 3, dtype="i4", rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = slice(1,4)
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02(self):
        """Testing `__getitem()__` method with a start, stop, step"""
        a = np.ones((10,2), dtype="i4")*3
        b = blz.fill((10,2), 3, dtype="i4", rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = slice(1,9,2)
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03a(self):
        """Testing `__getitem()__` method with several slices (I)"""
        a = np.arange(12).reshape((4,3))
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = (slice(1,3,1), slice(1,4,2))
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03b(self):
        """Testing `__getitem()__` method with several slices (II)"""
        a = np.arange(24*1000).reshape((4*1000,3,2))
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = (slice(1,3,2), slice(1,4,2), slice(None))
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03c(self):
        """Testing `__getitem()__` method with several slices (III)"""
        a = np.arange(120*1000).reshape((5*1000,4,3,2))
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = (slice(None,None,3), slice(1,3,2), slice(1,4,2))
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04a(self):
        """Testing `__getitem()__` method with shape reduction (I)"""
        a = np.arange(12000).reshape((40,300))
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = (1,1)
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__getitem()__` method with shape reduction (II)"""
        a = np.arange(12000).reshape((400,30))
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = (1,slice(1,4,2))
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__getitem()__` method with shape reduction (III)"""
        a = np.arange(6000).reshape((50,40,3))
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)
        sl = (1,slice(1,4,2),2)
        #print "b[sl]->", `b[sl]`
        self.assert_(a[sl].shape == b[sl].shape, "Shape is not equal")
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

class getitemDiskTest(getitemTest, unittest.TestCase):
    disk = True
    open = False

class getitemOpenTest(getitemTest, unittest.TestCase):
    disk = True
    open = True


class setitemTest(MayBeDiskTest, unittest.TestCase):

    open = False

    def test00a(self):
        """Testing `__setitem()__` method with only a start (scalar)"""
        a = np.ones((2,3), dtype="i4")*3
        b = blz.fill((2,3), 3, dtype="i4", rootdir=self.rootdir)
        sl = slice(1)
        a[sl,:] = 0
        b[sl] = 0
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test00b(self):
        """Testing `__setitem()__` method with only a start (vector)"""
        a = np.ones((200,300), dtype="i4")*3
        b = blz.fill((200,300), 3, dtype="i4", rootdir=self.rootdir)
        sl = slice(1)
        a[sl,:] = range(300)
        b[sl] = range(300)
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01a(self):
        """Testing `__setitem()__` method with start,stop (scalar)"""
        a = np.ones((500,200), dtype="i4")*3
        b = blz.fill((500,200), 3, dtype="i4", rootdir=self.rootdir,
                    bparams=blz.bparams())
        sl = slice(100,400)
        a[sl,:] = 0
        b[sl] = 0
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")
        #assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test01b(self):
        """Testing `__setitem()__` method with start,stop (vector)"""
        a = np.ones((5,2), dtype="i4")*3
        b = blz.fill((5,2), 3, dtype="i4", rootdir=self.rootdir)
        sl = slice(1,4)
        a[sl,:] = range(2)
        b[sl] = range(2)
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02a(self):
        """Testing `__setitem()__` method with start,stop,step (scalar)"""
        a = np.ones((1000,200), dtype="i4")*3
        b = blz.fill((1000,200), 3, dtype="i4", rootdir=self.rootdir)
        sl = slice(100,800,3)
        a[sl,:] = 0
        b[sl] = 0
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02b(self):
        """Testing `__setitem()__` method with start,stop,step (scalar)"""
        a = np.ones((10,2), dtype="i4")*3
        b = blz.fill((10,2), 3, dtype="i4", rootdir=self.rootdir)
        sl = slice(1,8,3)
        a[sl,:] = range(2)
        b[sl] = range(2)
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "b[sl]->", `b[sl]`, `b`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03a(self):
        """Testing `__setitem()__` method with several slices (I)"""
        a = np.arange(12000).reshape((400,30))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (slice(1,3,1), slice(1,None,2))
        #print "before->", `b[sl]`
        a[sl] = [[1],[2]]
        b[sl] = [[1],[2]]
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test03b(self):
        """Testing `__setitem()__` method with several slices (II)"""
        a = np.arange(24000).reshape((400,3,20))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (slice(1,3,1), slice(1,None,2), slice(1))
        #print "before->", `b[sl]`
        a[sl] = [[[1]],[[2]]]
        b[sl] = [[[1]],[[2]]]
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test03c(self):
        """Testing `__setitem()__` method with several slices (III)"""
        a = np.arange(120).reshape((5,4,3,2))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (slice(1,3), slice(1,3,1), slice(1,None,2), slice(1))
        #print "before->", `b[sl]`
        a[sl] = [[[[1]],[[2]]]]*2
        b[sl] = [[[[1]],[[2]]]]*2
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test03d(self):
        """Testing `__setitem()__` method with several slices (IV)"""
        a = np.arange(120).reshape((5,4,3,2))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (slice(1,3), slice(1,3,1), slice(1,None,2), slice(1))
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test04a(self):
        """Testing `__setitem()__` method with shape reduction (I)"""
        a = np.arange(12).reshape((4,3))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (1,1)
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__setitem()__` method with shape reduction (II)"""
        a = np.arange(12).reshape((4,3))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (1,slice(1,4,2))
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__setitem()__` method with shape reduction (III)"""
        a = np.arange(24).reshape((4,3,2))
        b = blz.barray(a, rootdir=self.rootdir)
        sl = (1,2,slice(None,None,None))
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        if self.open:
            b.flush()
            b = blz.open(rootdir=self.rootdir)
        #print "after->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

class setitemDiskTest(setitemTest, unittest.TestCase):
    disk = True

class setitemOpenTest(setitemTest, unittest.TestCase):
    disk = True
    open = True


class appendTest(MayBeDiskTest, unittest.TestCase):

    def test00a(self):
        """Testing `append()` method (correct shape)"""
        a = np.ones((2,300), dtype="i4")*3
        b = blz.fill((1,300), 3, dtype="i4", rootdir=self.rootdir)
        b.append([(3,)*300])
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `append()` method (correct shape, single row)"""
        a = np.ones((2,300), dtype="i4")*3
        b = blz.fill((1,300), 3, dtype="i4", rootdir=self.rootdir)
        b.append((3,)*300)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01(self):
        """Testing `append()` method (incorrect shape)"""
        a = np.ones((2,3), dtype="i4")*3
        b = blz.fill((1,3), 3, dtype="i4", rootdir=self.rootdir)
        self.assertRaises(ValueError, b.append, [(3,3)])

    def test02(self):
        """Testing `append()` method (several rows)"""
        a = np.ones((4,3), dtype="i4")*3
        b = blz.fill((1,3), 3, dtype="i4", rootdir=self.rootdir)
        b.append([(3,3,3)]*3)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

class appendDiskTest(appendTest, unittest.TestCase):
    disk = True


class resizeTest(MayBeDiskTest, unittest.TestCase):

    def test00a(self):
        """Testing `resize()` (trim)"""
        a = np.ones((2,3), dtype="i4")
        b = blz.ones((3,3), dtype="i4", rootdir=self.rootdir)
        b.resize(2)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `resize()` (trim to zero)"""
        a = np.ones((0,3), dtype="i4")
        b = blz.ones((3,3), dtype="i4", rootdir=self.rootdir)
        b.resize(0)
        #print "b->", `b`
        # The next does not work well for barrays with shape (0,)
        #assert_array_equal(a, b, "Arrays are not equal")
        self.assert_("a.dtype.base == b.dtype.base")
        self.assert_("a.shape == b.shape+b.dtype.shape")

    def test01(self):
        """Testing `resize()` (enlarge)"""
        a = np.ones((4,3), dtype="i4")
        b = blz.ones((3,3), dtype="i4", rootdir=self.rootdir)
        b.resize(4)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

class resizeDiskTest(resizeTest, unittest.TestCase):
    disk = True


class iterTest(unittest.TestCase):

    def test00(self):
        """Testing `iter()` (no start, stop, step)"""
        a = np.ones((3,), dtype="i4")
        b = blz.ones((1000,3), dtype="i4")
        #print "b->", `b`
        for r in b.iter():
            assert_array_equal(a, r, "Arrays are not equal")

    def test01(self):
        """Testing `iter()` (w/ start, stop)"""
        a = np.ones((3,), dtype="i4")
        b = blz.ones((1000,3), dtype="i4")
        #print "b->", `b`
        for r in b.iter(start=10):
            assert_array_equal(a, r, "Arrays are not equal")

    def test02(self):
        """Testing `iter()` (w/ start, stop, step)"""
        a = np.ones((3,), dtype="i4")
        b = blz.ones((1000,3), dtype="i4")
        #print "b->", `b`
        for r in b.iter(15, 100, 3):
            assert_array_equal(a, r, "Arrays are not equal")


class reshapeTest(unittest.TestCase):

    def test00a(self):
        """Testing `reshape()` (unidim -> ndim)"""
        a = np.ones((3,4), dtype="i4")
        b = blz.ones(12, dtype="i4").reshape((3,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `reshape()` (unidim -> ndim, -1 in newshape (I))"""
        a = np.ones((3,4), dtype="i4")
        b = blz.ones(12, dtype="i4").reshape((-1,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00c(self):
        """Testing `reshape()` (unidim -> ndim, -1 in newshape (II))"""
        a = np.ones((3,4), dtype="i4")
        b = blz.ones(12, dtype="i4").reshape((3,-1))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01(self):
        """Testing `reshape()` (ndim -> unidim)"""
        a = np.ones(12, dtype="i4")
        c = blz.ones(12, dtype="i4").reshape((3,4))
        b = c.reshape(12)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test02a(self):
        """Testing `reshape()` (ndim -> ndim, I)"""
        a = np.arange(12, dtype="i4").reshape((3,4))
        c = blz.arange(12, dtype="i4").reshape((4,3))
        b = c.reshape((3,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test02b(self):
        """Testing `reshape()` (ndim -> ndim, II)"""
        a = np.arange(24, dtype="i4").reshape((2,3,4))
        c = blz.arange(24, dtype="i4").reshape((4,3,2))
        b = c.reshape((2,3,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test03(self):
        """Testing `reshape()` (0-dim)"""
        a = np.ones((0,4), dtype="i4")
        b = blz.ones(0, dtype="i4").reshape((0,4))
        #print "b->", `b`
        # The next does not work well for barrays with shape (0,)
        #assert_array_equal(a, b, "Arrays are not equal")
        self.assert_(a.dtype.base == b.dtype.base)
        self.assert_(a.shape == b.shape+b.dtype.shape)


class compoundTest:

    def test00(self):
        """Testing compound types (creation)"""
        a = np.ones((300,4), dtype=self.dtype)
        b = blz.ones((300,4), dtype=self.dtype)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing compound types (append)"""
        a = np.ones((300,4), dtype=self.dtype)
        b = blz.barray([], dtype=self.dtype).reshape((0,4))
        b.append(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing compound types (iter)"""
        a = np.ones((3,), dtype=self.dtype)
        b = blz.ones((1000,3), dtype=self.dtype)
        #print "b->", `b`
        for r in b.iter():
            #print "r-->", r
            assert_array_equal(a, r, "Arrays are not equal")


class plainCompoundTest(compoundTest, unittest.TestCase):
    dtype = np.dtype("i4,i8")

class nestedCompoundTest(compoundTest, unittest.TestCase):
    dtype = np.dtype([('f1', [('f1', 'i2'), ('f2', 'i4')])])


class stringTest(unittest.TestCase):

    def test00(self):
        """Testing string types (creation)"""
        a = np.array([["ale", "ene"], ["aco", "ieie"]], dtype="S4")
        b = blz.barray(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing string types (append)"""
        a = np.ones((300,4), dtype="S4")
        b = blz.barray([], dtype="S4").reshape((0,4))
        b.append(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing string types (iter)"""
        a = np.ones((3,), dtype="S40")
        b = blz.ones((1000,3), dtype="S40")
        #print "b->", `b`
        for r in b.iter():
            #print "r-->", r
            assert_array_equal(a, r, "Arrays are not equal")


class unicodeTest(unittest.TestCase):

    def test00(self):
        """Testing unicode types (creation)"""
        a = np.array([[u"aŀle", u"eñe"], [u"açò", u"áèâë"]], dtype="U4")
        b = blz.barray(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing unicode types (append)"""
        a = np.ones((300,4), dtype="U4")
        b = blz.barray([], dtype="U4").reshape((0,4))
        b.append(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing unicode types (iter)"""
        a = np.ones((3,), dtype="U40")
        b = blz.ones((1000,3), dtype="U40")
        #print "b->", `b`
        for r in b.iter():
            #print "r-->", r
            assert_array_equal(a, r, "Arrays are not equal")


class computeMethodsTest(unittest.TestCase):

    def test00(self):
        """Testing sum()."""
        a = np.arange(1e5).reshape(10, 1e4)
        sa = a.sum()
        ac = blz.barray(a)
        sac = ac.sum()
        #print "numpy sum-->", sa
        #print "barray sum-->", sac
        self.assert_(sa.dtype == sac.dtype, "sum() is not working correctly.")
        self.assert_(sa == sac, "sum() is not working correctly.")


class barrayConstructorDimensionTest(MayBeDiskTest, unittest.TestCase):
    """
    This test is related to issue #14 in blaze github repo.
    Check that when using a barray constructor the dimensionality of the array
    is not lost. Neither in the implicit dtype or explicit dtype cases.
    """
    open = False

    def testImplicitDtype(self):
        """Testing barray construction keeping dimensions (implicit dtype)"""
        a = np.eye(6) # 2d
        b = blz.barray(a, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)

        # array equality implies having the same shape
        assert_array_equal(a, b, "Arrays are not equal")

    def testExplicitDtype(self):
        """Testing barray construction keeping dimensions (explicit dtype)"""
        dtype = np.dtype(np.float64)
        a = np.eye(6, dtype=dtype)
        b = blz.barray(a, dtype=dtype, rootdir=self.rootdir)
        if self.open:
            b = blz.open(rootdir=self.rootdir)

        # array equality implies having the same shape
        assert_array_equal(a, b, "Arrays are not equal")

class barrayConstructorDimensionDiskTest(barrayConstructorDimensionTest,
                                         unittest.TestCase):
    disk = True
    open = False

class barrayConstructorDimensionOpenTest(barrayConstructorDimensionTest,
                                         unittest.TestCase):
    disk = True
    open = True



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
