from __future__ import absolute_import, division, print_function

from blaze.json import resource
from blaze.utils import example

def test_simple():
    assert len(list(resource(example('accounts.json')))) == 5

def test_streaming():
    assert len(list(resource(example('accounts-streaming.json')))) == 5

def test_simple_gzip():
    assert len(list(resource(example('accounts.json.gz')))) == 5

def test_streaming_gzip():
    assert len(list(resource(example('accounts-streaming.json.gz')))) == 5
