from __future__ import absolute_import, division, print_function

import pytest
import os
import json
from datetime import datetime, timedelta

import pandas as pd
from pytz import utc

from blaze.server.serialization import (json_dumps, json_dumps_trusted,
                                        object_hook, object_hook_trusted)


@pytest.mark.parametrize('serializers', [(json_dumps, object_hook),
                                          (json_dumps_trusted, object_hook_trusted)])
@pytest.mark.parametrize('input_,serialized', (
    ([1, datetime(2000, 1, 1, 12, 30, 0, 0, utc)],
     '[1, {"__!datetime": "2000-01-01T12:30:00+00:00"}]'),
    ([1, pd.NaT], '[1, {"__!datetime": "NaT"}]'),
    ([1, frozenset([1, 2, 3])], '[1, {"__!frozenset": [1, 2, 3]}]'),
    ([1, timedelta(seconds=5)], '[1, {"__!timedelta": 5.0}]'),
))
def test_json_encoder(serializers, input_, serialized):
    serializer, deserializer = serializers
    result = json.dumps(input_, default=serializer)
    assert result == serialized
    assert json.loads(result, object_hook=deserializer) == input_
