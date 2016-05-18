from __future__ import absolute_import, division, print_function

import os
from blaze.utils import tmpfile


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)
