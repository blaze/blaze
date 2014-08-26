from blaze.api.resource import *
from blaze.data import *

def test_resource_csv():
    assert isinstance(resource('blaze/api/tests/a1.csv'), CSV)
