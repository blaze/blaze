import os
import pytest

from blaze.utils import tmpfile, suppress


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)


def test_suppress():
    class MyTypeError(TypeError):
        pass

    class MyExc(Exception):
        pass

    with pytest.raises(MyExc):
        raise MyExc('asdf')

    with suppress(MyExc):
        raise MyExc('asdf')

    with pytest.raises(TypeError):
        raise MyTypeError('asdf')

    with suppress(TypeError):
        raise MyTypeError('asdf')
