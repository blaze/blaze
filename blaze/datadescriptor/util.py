
from contextlib import contextmanager
import tempfile
import os

from dynd import nd

@contextmanager
def filetext(text, extension=''):
    # write text to hidden file
    handle, filename = tempfile.mkstemp(extension)
    with openfile(extension=extension) as filename:
        with open(filename, "w") as f:
            f.write(text)

        yield filename

@contextmanager
def openfile(extension=''):
    filename = tempfile.mktemp()

    yield filename

    os.remove(filename)


def validate(schema, item):
    try:
        nd.array(item, dtype=schema)
        return True
    except:
        return False

def coerce(schema, item):
    return nd.as_py(nd.array(item, dtype=schema))


def raises(err, lamda):
    try:
        lamda()
        return False
    except err:
        return True


def coerce_record_to_row(schema, rec):
    """

    >>> from datashape import dshape

    >>> schema = dshape('{x: int, y: int}')
    >>> coerce_record_to_row(schema, {'x': 1, 'y': 2})
    [1, 2]
    """
    return [rec[name] for name in schema[0].names]


def coerce_row_to_dict(schema, row):
    """

    >>> from datashape import dshape

    >>> schema = dshape('{x: int, y: int}')
    >>> coerce_row_to_dict(schema, (1, 2)) # doctest: +SKIP
    {'x': 1, 'y': 2}
    """
    return dict((name, item) for name, item in zip(schema[0].names, row))
