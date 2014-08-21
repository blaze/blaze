import os
import pytest
import numpy as np
from sqlalchemy import create_engine
from blaze import discover
from blaze.data.drop import drop, HDF5, SQL, CSV
from blaze.data.drop import Collection


data = [(1, 32.4, 'Alice'),
        (2, 234.24, 'Bob'),
        (4, -430.0, 'Joe')]


x = np.array(data, dtype=[('id', int), ('amount', float), ('name', '|S5')])


schema = discover(x).subshape[0]


class TestDrop(object):
    def test_hdf5(self, h):
        h = HDF5('test.h5', '/test', schema=schema)
        h.extend(data)
        drop(h)
        with pytest.raises(Exception):
            pass

    def test_sql(self):
        engine = create_engine('sqlite:///:memory:')
        sql = SQL(engine, 'test', schema=schema)
        sql.extend(data)
        drop(sql)

    def test_csv(self):
        csv = CSV('test.csv', schema=schema)
        csv.extend(data)
        drop(csv)
        assert not os.path.exists('test.csv')

    def test_mongo(self):
        pass
