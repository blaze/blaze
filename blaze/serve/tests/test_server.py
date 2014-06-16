from blaze.serve.server import app, datasets
from blaze.data.python import Python

accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

datasets['accounts'] = accounts
datasets['cities'] = cities

def test_basic():
    test = app.test_client()
    assert 'OK' in test.get('/').status


def test_datasets():
    test = app.test_client()
    response = test.get('/datasets.json')
    assert 'accounts' in response.data
    assert 'cities' in response.data

    assert str(accounts.dshape) in response.data
    assert str(cities.dshape) in response.data
