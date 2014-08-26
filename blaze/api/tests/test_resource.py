from blaze.api.resource import *
from blaze.data import *
from blaze.api.into import into

def test_resource_csv():
    assert isinstance(resource('blaze/api/tests/a1.csv'), CSV)

def test_into_resource():
    assert into(list, 'blaze/api/tests/a1.csv')
