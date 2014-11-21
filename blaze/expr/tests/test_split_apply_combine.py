from blaze.expr import by, symbol

def test_by_raises_informative_error_on_old_syntax():
    s = symbol('t', 'var * {name: string, amount: int}')
    try:
        by(s.name, s.amount.sum())
        assert False
    except ValueError as e:
        assert 'please' in str(e).lower()
