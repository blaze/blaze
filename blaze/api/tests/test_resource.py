from blaze.api.resource import *
from blaze.data import *
from blaze.api.into import into

def test_resource_csv():
    assert isinstance(resource('blaze/api/tests/accounts_1.csv'), CSV)

def test_into_resource():
    assert into(list, 'blaze/api/tests/accounts_1.csv') == [(1, 'Alice', 100),
                                                            (2, 'Bob', 200)]

def test_into_directory_of_csv_files():
    assert into(list, 'blaze/api/tests/accounts_*.csv') == [(1, 'Alice', 100),
                                                            (2, 'Bob', 200),
                                                            (3, 'Charlie', 300),
                                                            (4, 'Dan', 400),
                                                            (5, 'Edith', 500)]
