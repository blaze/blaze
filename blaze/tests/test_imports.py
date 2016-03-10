import blaze

def test_basic_interface():
    blaze.compute
    blaze.into
    blaze.discover

def test_no_leeking():
    # unique is a commonly used toolz function and a good example of something
    # that shouldn't leak out to the main namespace
    assert not hasattr(blaze, 'unique')
