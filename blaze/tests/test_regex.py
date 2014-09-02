from blaze.regex import RegexDispatcher
import re

def test_regex_dispatcher():
    foo = RegexDispatcher('foo')

    @foo.register('\d*')
    def a(s):
        return int(s)

    @foo.register('\D*')
    def b(s):
        return s

    @foo.register('0\d*', priority=11)
    def c(s):
        return s

    assert set(foo.funcs.values()) == set([a, b, c])
    assert foo.dispatch('123') == a
    assert foo.dispatch('hello') == b
    assert foo.dispatch('0123') == c


    assert foo('123') == 123
    assert foo('hello') == 'hello'
    assert foo('0123') == '0123'
