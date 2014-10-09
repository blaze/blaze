from __future__ import absolute_import, division, print_function

import os
from collections import Iterator
from datetime import datetime
import pytest

import datashape
from datashape import dshape, Record

from blaze.expr.table import TableSymbol
from blaze.compute.python import compute
from blaze import Table, into, discover, DataDescriptor
from blaze.compatibility import min_python_version, xfail, PY3
from blaze.data import CSV
from blaze.utils import filetext, tmpfile, example
from blaze.data.utils import tuplify
from blaze.data.csv import drop, has_header, discover_dialect
from dynd import nd


def sanitize(lines):
    return '\n'.join(line.strip() for line in lines.split('\n'))


def test_schema_detection_modifiers():
    text = "name amount date\nAlice 100 20120101\nBob 200 20120102"
    with filetext(text) as fn:
        assert (CSV(fn).schema ==
                dshape('{name: string, amount: ?int64, date: ?int64}'))
        assert (CSV(fn, columns=['NAME', 'AMOUNT', 'DATE']).schema ==
                dshape('{NAME: string, AMOUNT: ?int64, DATE: ?int64}'))
        assert (str(CSV(fn, types=['string', 'int32', 'date']).schema) ==
                str(dshape('{name: string, amount: int32, date: date}')))

        a = CSV(fn, typehints={'date': 'date'}).schema
        b = dshape('{name: string, amount: ?int64, date: date}')
        assert str(a) == str(b)


def test_homogenous_schema():
    with filetext("1,1\n2,2\n3,3") as fn:
        assert (CSV(fn, columns=['x', 'y']).schema ==
                dshape('{x: int64, y: int64}'))


def test_a_mode():
    text = ("id, name, balance\n1, Alice, 100\n2, Bob, 200\n"
            "3, Charlie, 300\n4, Denis, 400\n5, Edith, 500")
    with filetext(text) as fn:
        csv = CSV(fn, 'a')
        csv.extend([(6, 'Frank', 600),
                    (7, 'Georgina', 700)])

        result = set(csv[:, 'name'])
        assert 'Georgina' in result


def test_sep_kwarg():
    csv = CSV('foo', 'w', sep=';', schema='{x: int, y: int}')
    assert csv.dialect['delimiter'] == ';'


def test_columns():
    # This is really testing the core interface
    dd = CSV('foo', 'w', schema='{name: string, amount: int}')
    assert list(dd.columns) == ['name', 'amount']


def test_unicode():
    this_dir = os.path.dirname(__file__)
    filename = os.path.join(this_dir, 'unicode.csv')
    dd = CSV(filename, columns=['a', 'b'], encoding='utf8')
    assert dd.schema == dshape('{a: string, b: ?int64}')
    assert dd[0]


@pytest.fixture
def buf():
    b = sanitize(
    u"""Name Amount
        Alice 100
        Bob 200
        Alice 50
    """)
    return b


data = (('Alice', 100),
        ('Bob', 200),
        ('Alice', 50))


@pytest.yield_fixture
def csv_file(buf):
    with filetext(buf, extension='.csv') as dialect_csv_file:
        yield dialect_csv_file


@pytest.fixture
def schema():
    return "{ name: string, amount: int }"


@pytest.fixture
def dd(csv_file, schema):
    return CSV(csv_file, dialect='excel', schema=schema, delimiter=' ',
               mode='r+')


def test_dynd(dd):
    assert isinstance(dd.dynd[0], nd.array)


def test_row(dd):
    assert tuplify(dd[0]) == ('Alice', 100)
    assert tuplify(dd[1]) == ('Bob', 200)


def test_rows(dd):
    assert tuplify(dd[[0, 1]]) == (('Alice', 100), ('Bob', 200))


def test_point(dd):
    assert dd[0, 0] == 'Alice'
    assert dd[1, 1] == 200


def test_nested(dd):
    assert tuplify(dd[[0, 1], 0]) == ('Alice', 'Bob')
    assert tuplify(dd[[0, 1], 1]) == (100, 200)
    assert tuplify(dd[0, [0, 1]]) == ('Alice', 100)
    assert tuplify(dd[[1, 0], [0, 1]]) == (('Bob', 200), ('Alice', 100))


def test_slices(dd):
    assert list(dd[:, 1]) == [100, 200, 50]
    assert list(dd[1:, 1]) == [200, 50]
    assert list(dd[0, :]) == ['Alice', 100]


def test_names(dd):
    assert list(dd[:, 'name']) == ['Alice', 'Bob', 'Alice']
    assert tuplify(dd[:, ['amount', 'name']]) == ((100, 'Alice'), (200, 'Bob'),
                                                  (50, 'Alice'))


def test_dynd_complex(dd):
    assert (tuplify(dd[:, ['amount', 'name']]) ==
            tuplify(nd.as_py(dd.dynd[:, ['amount', 'name']], tuple=True)))


def test_laziness(dd):
    assert isinstance(dd[:, 1], Iterator)


def test_schema_detection(csv_file):
    dd = CSV(csv_file)
    assert dd.schema == dshape('{Name: string, Amount: ?int64}')

    dd = CSV(csv_file, columns=['foo', 'bar'])
    assert dd.schema == dshape('{foo: string, bar: ?int64}')


@min_python_version
def test_has_header(buf):
    assert has_header(buf)


def test_overwrite_delimiter(dd):
    assert dd.dialect['delimiter'] == ' '


def test_content(dd):
    s = str(list(dd))
    assert 'Alice' in s and 'Bob' in s


def test_append(dd, csv_file):
    dd.extend([('Alice', 100)])
    with open(csv_file) as f:
        lines = f.readlines()
    assert lines[-1].strip() == 'Alice 100'


def test_append_dict(dd, csv_file):
    dd.extend([{'name': 'Alice', 'amount': 100}])
    with open(csv_file) as f:
        lines = f.readlines()

    assert lines[-1].strip() == 'Alice 100'


def test_extend_structured_newline():
    with filetext('1,1.0\n2,2.0\n') as fn:
        csv = CSV(fn, 'r+', schema='{x: int32, y: float32}', delimiter=',')
        csv.extend([(3, 3)])
        assert tuplify(tuple(csv)) == ((1, 1.0), (2, 2.0), (3, 3.0))

def test_tuple_types():
    """
    CSVs with uniform types still create record types with names
    """
    with filetext('1,1\n2,2\n') as fn:
        csv = CSV(fn, 'r+', delimiter=',')
        assert csv[0] == (1, 1)
        assert isinstance(csv.schema[0], Record)
        assert len(csv.schema[0].types) == 2
        assert len(set(csv.schema[0].types)) == 1


def test_extend_structured_no_newline():
    with filetext('1,1.0\n2,2.0') as fn:
        csv = CSV(fn, 'r+', schema='{x: int32, y: float32}', delimiter=',')
        csv.extend([(3, 3)])
        assert tuplify(tuple(csv)) == ((1, 1.0), (2, 2.0), (3, 3.0))


@xfail(reason="\n perceived as missing value.  Not allowed in int types")
def test_extend_structured_many_newlines():
    with filetext('1,1.0\n2,2.0\n\n\n\n') as fn:
        csv = CSV(fn, 'r+', schema='{x: int32, y: float32}', delimiter=',')
        csv.extend([(3, 3)])
        result = tuplify(tuple(csv))
        assert discover(result) == dshape('6 * (int64, float64)')


def test_discover_dialect():
    s = '1,1\r\n2,2'
    assert (discover_dialect(s) ==
            {'escapechar': None,
                'skipinitialspace': False,
                'quoting': 0,
                'delimiter': ',',
                'line_terminator': '\r\n',
                'quotechar': '"',
                'doublequote': False,
                'lineterminator': '\r\n',
                'sep': ','})
    assert (discover_dialect('1,1\n2,2') ==
            {'escapechar': None,
                'skipinitialspace': False,
                'quoting': 0,
                'delimiter': ',',
                'line_terminator': '\r\n',
                'quotechar': '"',
                'doublequote': False,
                'lineterminator': '\r\n',
                'sep': ','})


def test_errs_without_dshape(tmpcsv):
    with pytest.raises(ValueError):
        CSV(tmpcsv, 'w')


def test_creation(tmpcsv, schema):
    dd = CSV(tmpcsv, 'w', schema=schema, delimiter=' ')
    assert dd is not None


def test_creation_rw(tmpcsv, schema):
    dd = CSV(tmpcsv, 'w+', schema=schema, delimiter=' ')
    assert dd is not None


def test_append(tmpcsv, schema):
    dd = CSV(tmpcsv, 'w', schema=schema, delimiter=' ')
    dd.extend([data[0]])
    with open(tmpcsv) as f:
        s = f.readlines()[0].strip()
    assert s == 'Alice 100'


def test_extend(tmpcsv, schema):
    dd = CSV(tmpcsv, 'w', schema=schema, delimiter=' ')
    dd.extend(data)
    with open(tmpcsv) as f:
        lines = f.readlines()
    expected_lines = 'Alice 100', 'Bob 200', 'Alice 50'
    for i, eline in enumerate(expected_lines):
        assert lines[i].strip() == eline

    expected_dshape = datashape.DataShape(datashape.Var(),
                                          datashape.dshape(schema))

    assert str(dd.dshape) == str(expected_dshape)


def test_re_dialect():
    dialect1 = {'delimiter': ',', 'lineterminator': '\n'}
    dialect2 = {'delimiter': ';', 'lineterminator': '--'}

    text = '1,1\n2,2\n'

    schema = '{a: int32, b: int32}'

    with filetext(text) as source_fn:
        with filetext('') as dest_fn:
            src = CSV(source_fn, schema=schema, **dialect1)
            dst = CSV(dest_fn, mode='w', schema=schema, **dialect2)

            # Perform copy
            dst.extend(src)

            with open(dest_fn) as f:
                raw = f.read()
            assert raw == '1;1--2;2--'


def test_iter():
    with filetext('1,1\n2,2\n') as fn:
        dd = CSV(fn, schema='{a: int32, b: int32}')
        assert tuplify(list(dd)) == ((1, 1), (2, 2))


def test_chunks():
    with filetext('1,1\n2,2\n3,3\n4,4\n') as fn:
        dd = CSV(fn, schema='{a: int32, b: int32}')
        assert all(isinstance(chunk, nd.array) for chunk in dd.chunks())
        assert len(list(dd.chunks(blen=2))) == 2
        assert len(list(dd.chunks(blen=3))) == 2


def test_iter_structured():
    with filetext('1,2\n3,4\n') as fn:
        dd = CSV(fn, schema='{x: int, y: int}')
        assert tuplify(list(dd)) == ((1, 2), (3, 4))


@pytest.fixture
def kv_data():
    return (('k1', 'v1', 1, False),
            ('k2', 'v2', 2, True),
            ('k3', 'v3', 3, False))


@pytest.fixture
def kv_schema():
    return "{ f0: string, f1: string, f2: int16, f3: bool }"


@pytest.fixture
def kbuf():
    return sanitize(
    u"""k1,v1,1,False
        k2,v2,2,True
        k3,v3,3,False
    """)


@pytest.fixture
def kv_dd(tmpcsv, kbuf, kv_schema):
    with open(tmpcsv, 'w') as f:
        f.write(kbuf)
    return CSV(tmpcsv, schema=kv_schema)


def test_compute_kv(kv_dd, kv_schema):
    t = TableSymbol('t', kv_schema)
    lhs = compute(t['f2'].sum(), kv_dd)
    assert lhs == 1 + 2 + 3


def test_has_header_kv(kbuf):
    assert not has_header(kbuf)


def test_basic_object_type_kv(kv_dd):
    assert isinstance(kv_dd, DataDescriptor)
    assert isinstance(kv_dd.dshape.shape[0], datashape.Var)


def test_iter_kv(kv_dd, kv_data):
    assert tuplify(tuple(kv_dd)) == kv_data


def test_as_py_kv(kv_dd, kv_data):
    assert tuplify(kv_dd.as_py()) == kv_data


def test_getitem_start_kv(kv_dd, kv_data):
    assert (tuplify(kv_dd[0]) == kv_data[0])


def test_getitem_stop_kv(kv_dd, kv_data):
    assert tuplify(kv_dd[:1]) == kv_data[:1]


def test_getitem_step_kv(kv_dd, kv_data):
    assert tuplify(kv_dd[::2]) == kv_data[::2]


def test_getitem_start_step_kv(kv_dd, kv_data):
    assert tuplify(kv_dd[1::2]) == kv_data[1::2]


def test_repr_hdma():
    csv = CSV(example('hmda-small.csv'))
    t = TableSymbol('hmda', csv.schema)

    assert compute(t.head(), csv)

    columns = ['action_taken_name', 'agency_abbr', 'applicant_ethnicity_name']
    assert compute(t[columns].head(), csv)


@pytest.yield_fixture
def date_data():
    data = [('Alice', 100.0, datetime(2014, 9, 11, 0, 0, 0, 0)),
            ('Alice', -200.0, datetime(2014, 9, 10, 0, 0, 0, 0)),
            ('Bob', 300.0, None)]
    schema = dshape('{name: string, amount: float32, date: ?datetime}')
    with tmpfile('.csv') as f:
        csv = CSV(f, schema=schema, mode='w')
        csv.extend(data)
        yield CSV(f, schema=schema, mode='r')


def test_subset_with_date(date_data):
    csv = date_data
    sub = csv[[0, 1], 'date']
    expected = [datetime(2014, 9, 11, 0, 0, 0, 0),
                datetime(2014, 9, 10, 0, 0, 0, 0)]
    assert into(list, sub) == expected


def test_subset_no_date(date_data):
    csv = date_data
    expected = [(100.0, 'Alice'),
                (-200.0, 'Alice'),
                (300.0, 'Bob')]
    result = into(list, csv[:, ['amount', 'name']])
    assert result == expected


@pytest.yield_fixture
def csv(schema):
    csv = CSV('test.csv', schema=schema, mode='w')
    csv.extend(data)
    yield csv
    try:
        os.remove(csv.path)
    except OSError:
        pass


def test_drop(csv):
    assert os.path.exists(csv.path)
    drop(csv)
    assert not os.path.exists(csv.path)


@pytest.yield_fixture
def tmpcsv():
    with tmpfile('.csv') as f:
        yield f


def test_string_dataset(tmpcsv):
    raw = 'a,b,2.0\nc,1999,3.0\nd,3.0,4.0'
    with open(tmpcsv, mode='w') as f:
        f.write(raw)
    csv = CSV(tmpcsv, columns=list('xyz'))
    t = Table(csv)
    x = into(list, t)
    assert x == [('a', 'b', 2.0), ('c', '1999', 3.0), ('d', '3.0', 4.0)]
