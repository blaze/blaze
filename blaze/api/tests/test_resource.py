from blaze.api.resource import *
from blaze.data import *
from blaze.api.into import into
import os

dirname = os.path.dirname(__file__)

def test_resource_csv():
    fn = os.path.join(dirname, 'accounts_1.csv')
    assert isinstance(resource(fn), CSV)

def test_into_resource():
    fn = os.path.join(dirname, 'accounts_1.csv')
    assert into(list, fn) == [(1, 'Alice', 100),
                                                            (2, 'Bob', 200)]

def test_into_directory_of_csv_files():
    fns = os.path.join(dirname, 'accounts_*.csv')
    assert into(list, fns) == [(1, 'Alice', 100),
                                                            (2, 'Bob', 200),
                                                            (3, 'Charlie', 300),
                                                            (4, 'Dan', 400),
                                                            (5, 'Edith', 500)]
