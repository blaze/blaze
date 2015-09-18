from datetime import datetime
import os
import json

import pandas as pd
import pytest

from blaze.utils import tmpfile, json_dumps, object_hook


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)


@pytest.mark.parametrize('input_,serialized,deserliazied', (
    ([1, datetime(2000, 1, 1, 12, 30, 0)],
     '[1, {"__!datetime": "2000-01-01T12:30:00Z"}]',
     [1, pd.Timestamp("2000-01-01T12:30:00Z")]),
    ([1, frozenset([1, 2, 3])],
     '[1, {"__!frozenset": [1, 2, 3]}]',
     [1, frozenset([1, 2, 3])]),
))
def test_json_encoder(input_, serialized, deserliazied):
    result = json.dumps(input_, default=json_dumps)
    assert result == serialized
    assert json.loads(result, object_hook=object_hook) == deserliazied
