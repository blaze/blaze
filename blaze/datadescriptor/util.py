
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
