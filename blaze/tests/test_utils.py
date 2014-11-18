import os
import time

from blaze.utils import tmpfile, alongside


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)


def test_alongside():
    x = [0]

    def incx():
        x[0] = x[0] + 1

    with alongside(time.sleep, 100000):
        with alongside(incx):
            pass

    assert x[0] == 1
