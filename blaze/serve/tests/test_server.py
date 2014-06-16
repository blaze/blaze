from blaze.serve.server import app, datasets
from blaze.data.python import Python

dd = Python([['Alice', 100], ['Bob', 200]],
            schema='{name: string, amount: int32}')

datasets['accounts'] = dd

def test_basic():
    test = app.test_client()
    assert test.get('/')
