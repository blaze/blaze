from sqlalchemy import create_engine
from blaze.datadescriptor import SQL_DDesc, DyND_DDesc
from blaze.datadescriptor.util import raises
from dynd import nd
import unittest


class SingleTestClass(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:', echo=True)

    def tearDown(self):
        pass
        # How do I clean up an engine?

    def test_setup_with_uri(self):
        dd = SQL_DDesc('sqlite:///:memory:',
                       'accounts',
                       schema='{name: string, amount: int}')

    def test_can_connect(self):
        with self.engine.connect() as conn:
            assert not conn.closed
        assert conn.closed

    def test_table_creation(self):
        dd = SQL_DDesc(self.engine, 'testtable',
                               schema='{name: string, amount: int}',
                               primary_key='name')
        assert self.engine.has_table('testtable')


        assert dd.table.columns.get('name').primary_key
        assert not dd.table.columns.get('amount').primary_key
        assert dd.dshape == 'var * {name: string, amount: int}'

        assert raises(ValueError, lambda: SQL_DDesc(self.engine, 'testtable2'))


    def test_extension(self):
        dd = SQL_DDesc(self.engine, 'testtable2',
                               schema='{name: string, amount: int32}',
                               primary_key='name')

        data_list = [('Alice', 100), ('Bob', 50)]
        data_dict = [{'name': name, 'amount': amount} for name, amount in data_list]

        dd.extend(data_dict)

        with self.engine.connect() as conn:
            results = conn.execute('select * from testtable2')
            self.assertEquals(list(results), data_list)


        assert list(iter(dd)) == data_list or list(iter(dd)) == data_dict


    def test_chunks(self):
        schema = '{name: string, amount: int32}'
        dd = SQL_DDesc(self.engine, 'testtable3',
                                   schema=schema,
                                   primary_key='name')

        data_list = [('Alice', 100), ('Bob', 50), ('Charlie', 200)]
        data_dict = [{'name': name, 'amount': amount} for name, amount in data_list]
        chunk = nd.array(data_list, dtype=dd.dshape)

        dd.extend_chunks([chunk])

        assert list(iter(dd)) == data_list or list(iter(dd)) == data_dict

        self.assertEquals(len(list(dd.iterchunks(blen=2))), 2)
