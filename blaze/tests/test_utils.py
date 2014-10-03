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

    def raiser():
        raise MyExc('asdf')

    def te_raiser():
        raise MyTypeError('asdf')

    with pytest.raises(MyExc):
        raiser()

    with suppress(MyExc):
        raiser()

    with pytest.raises(TypeError):
        te_raiser()

    with suppress(TypeError):
        te_raiser()
