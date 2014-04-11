from sqlalchemy import create_engine
from blaze.datadescriptor.sql_data_descriptor import SQL_DDesc
from blaze.datadescriptor.util import raises

engine = create_engine('sqlite:///:memory:', echo=True)

def test_can_connect():
    with engine.connect() as conn:
        assert not conn.closed

def test_table_creation():
    dd = SQL_DDesc(engine, 'testtable',
                           schema='{name: string, amount: int}',
                           primary_key='name')
    assert engine.has_table('testtable')


    assert dd.table.columns.get('name').primary_key
    assert not dd.table.columns.get('amount').primary_key
    assert dd.dshape == 'var * {name: string, amount: int}'

    assert raises(ValueError, lambda: SQL_DDesc(engine, 'testtable2'))
