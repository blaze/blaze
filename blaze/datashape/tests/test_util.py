import unittest

import blaze
from blaze import datashape
from blaze import dshape

class TestDataShapeUtil(unittest.TestCase):
    def test_cat_dshapes(self):
        # concatenating 1 dshape is a no-op
        dslist = [dshape('3, 10, int32')]
        self.assertEqual(datashape.cat_dshapes(dslist),
                        dslist[0])
        # two dshapes
        dslist = [dshape('3, 10, int32'),
                        dshape('7, 10, int32')]
        self.assertEqual(datashape.cat_dshapes(dslist),
                        dshape('10, 10, int32'))

    def test_cat_dshapes_errors(self):
        # need at least one dshape
        self.assertRaises(ValueError, datashape.cat_dshapes, [])
        # dshapes need to match after the first dimension
        self.assertRaises(ValueError, datashape.cat_dshapes,
                        [dshape('3, 10, int32'), dshape('3, 1, int32')])

    def test_broadcastable(self):
        dslist = [dshape('10,20,30,int32'),
                        dshape('20,30,int32'), dshape('int32')]
        outshape = datashape.broadcastable(dslist, ranks=[1,1,0])
        self.assertEqual(outshape, (10,20))

        dslist = [dshape('10,20,30,40,int32'),
                        dshape('20,30,20,int32'), dshape('int32')]
        outshape = datashape.broadcastable(dslist, ranks=[1,1,0])
        self.assertEqual(outshape, (10,20,30))

        dslist = [dshape('10,20,30,40,int32'),
                        dshape('20,30,40,int32'), dshape('int32')]
        outshape = datashape.broadcastable(dslist, ranks=[1,1,0],
                        rankconnect=[set([(0,0),(1,0)])])
        self.assertEqual(outshape, (10,20,30))

if __name__ == '__main__':
    unittest.main()

