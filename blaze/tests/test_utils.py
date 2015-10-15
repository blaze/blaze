from datetime import datetime, timedelta
import os
import json

import pandas as pd
import pytest
from pytz import utc

from blaze.utils import tmpfile, json_dumps, object_hook


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)


@pytest.mark.parametrize('input_,serialized', (
    ([1, datetime(2000, 1, 1, 12, 30, 0, 0, utc)],
     '[1, {"__!datetime": "2000-01-01T12:30:00+00:00"}]'),
    ([1, pd.NaT], '[1, {"__!datetime": "NaT"}]'),
    ([1, frozenset([1, 2, 3])], '[1, {"__!frozenset": [1, 2, 3]}]'),
    ([1, timedelta(seconds=5)], '[1, {"__!timedelta": 5.0}]'),
))
def test_json_encoder(input_, serialized):
    result = json.dumps(input_, default=json_dumps)
    assert result == serialized
    assert json.loads(result, object_hook=object_hook) == input_
