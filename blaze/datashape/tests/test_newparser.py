from blaze.datashape.parser import *
from unittest import skip

def assert_parse(self, src, lexed):
    src = src.replace('\\n', '\n')
    src = src.replace('\\r', '\r')

    l = lex(src)
    r = []
    for t in l:
        r.append(str(t))
    act = ' '.join(r)

    if act != lexed:
        print('Actual:  ', act)
        print('Expected:', lexed)
    assert_equal(act, lexed)


def test_all_the_strings():
    parse('a')
    parse('a, b')
    parse('a, b, c')
    parse('a,      b')
    parse('a,      b  ,     d')
    parse('foo = c')
    parse('a b c = d,e,f')
    parse('   a b c = d, e   ')
    parse('foo a b = c, d')
    parse('foo a b = c,d,e')
    parse('foo a b = c,d,e,   f')
    parse('foo a b = c,   d,   e,   f')
    parse('foo b = c,   d,   e,   f')
    parse('a b c = d, e')
    parse('a b c = bar, foo')
    parse('800, 600, RGBA')
    parse('Pixel A = A')
    parse('Pixel A = A, B')
    parse('Pixel A = 800, 600, A, A')
    parse('Type A B = 800, 600, A, B')
    parse('Type A B = {}')
    parse('Type A B = {A:B}')
    parse('''
    Type = {
        A: B;
        C: D;
        E: (A,B)
    }
    ''')

    parse('''
    Type a b = {
        A : B;
        C : D;
        E : (A,B)
    }
    ''')

    parse('Type = {} ')

    parse('''
    Stock = {
      name   : string;
      min    : int64;
      max    : int64;
      mid    : int64;
      volume : float;
      close  : float;
      open   : float;
    }
    ''')

def test_trailing_semi():
    a = parse('''
    Type = {
        a: int;
        b: float;
        c: (int,int)
    }
    ''')

    b = parse('''
    Type = {
        a: int;
        b: float;
        c: (int,int);
    }
    ''')

    assert a == b
