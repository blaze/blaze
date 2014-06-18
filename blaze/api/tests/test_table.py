from blaze.api.table import Table, compute


data = (('Alice', 100),
        ('Bob', 200))

t = Table(data)

def test_resources():
    assert t.resources() == {t: t.data}


def test_compute():
    assert compute(t) == data
