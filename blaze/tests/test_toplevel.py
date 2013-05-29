import os.path

from blaze.test_utils import temp_dir

import blaze.toplevel as toplevel
from blaze.params import params
from blaze import dshape
from blaze.sources.chunked import BArraySource, BTableSource
from blaze.eclass import eclass

def test_open_carray():
    with temp_dir() as temp:
        # Create an array on disk
        array_filename = os.path.join(temp, 'carray')
        p = params(storage=array_filename)
        ds = dshape('1,int32')
        a = BArraySource([2], dshape=ds, params=p)
        del a

        # Open array with open function
        uri = 'carray://' + array_filename
        c = toplevel.open(uri)
        assert c.datashape == ds

        # Test delayed mode
        c = toplevel.open(uri, eclass=eclass.delayed)
        assert c.datashape == ds

def test_open_ctable():
    with temp_dir() as temp:
        # Create an table on disk
        table_filename = os.path.join(temp, 'ctable')
        p = params(storage=table_filename)
        ds = dshape('1,{ x: int32; y: int32 }')
        t = BTableSource(data=[(1, 1), (2, 2)], dshape=ds, params=p)
        del t

        # Open table with open function
        uri = 'ctable://' + table_filename
        c = toplevel.open(uri)
        assert c.datashape == ds

        # Test delayed mode
        c = toplevel.open(uri, eclass=eclass.delayed)
        assert c.datashape == ds
