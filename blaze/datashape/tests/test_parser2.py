from blaze.datashape.parser import *
from unittest import skip

ex1 = tyinst(conargs=('a',))
ex2 = tyinst(conargs=('a', 'b'))
ex3 = tyinst(conargs=('a', 'b', 'c'))
ex4 = tyinst(conargs=('a', 'b'))
ex5 = tyinst(conargs=('a', 'b', 'd'))
ex6 = tydecl(lhs=simpletype(nargs=0, tycon='foo', tyvars=()), rhs=('c',))
ex7 = tydecl(lhs=simpletype(nargs=2, tycon='a', tyvars=('b', 'c')), rhs=('d', 'e', 'f'))
ex8 = tydecl(lhs=simpletype(nargs=2, tycon='a', tyvars=('b', 'c')), rhs=('d', 'e'))
ex9 = tydecl(lhs=simpletype(nargs=2, tycon='foo', tyvars=('a', 'b')), rhs=('c', 'd'))
ex10 = tydecl(lhs=simpletype(nargs=2, tycon='foo', tyvars=('a', 'b')), rhs=('c', 'd', 'e'))
ex11 = tydecl(lhs=simpletype(nargs=2, tycon='foo', tyvars=('a', 'b')), rhs=('c', 'd', 'e', 'f'))
ex12 = tydecl(lhs=simpletype(nargs=2, tycon='foo', tyvars=('a', 'b')), rhs=('c', 'd', 'e', 'f'))
ex13 = tydecl(lhs=simpletype(nargs=1, tycon='foo', tyvars=('b',)), rhs=('c', 'd', 'e', 'f'))
ex13 = tydecl(lhs=simpletype(nargs=2, tycon='a', tyvars=('b', 'c')), rhs=('d', 'e'))
ex14 = tydecl(lhs=simpletype(nargs=2, tycon='a', tyvars=('b', 'c')), rhs=('bar', 'foo'))
ex15 = tyinst(conargs=(800, 600, 'RGBA'))
ex16 = tydecl(lhs=simpletype(nargs=1, tycon='Pixel', tyvars=('A',)), rhs=('A',))
ex17 = tydecl(lhs=simpletype(nargs=1, tycon='Pixel', tyvars=('A',)), rhs=('A', 'B'))
ex18 = tydecl(lhs=simpletype(nargs=1, tycon='Pixel', tyvars=('A',)), rhs=(800, 600, 'A', 'A'))
ex19 = tydecl(lhs=simpletype(nargs=2, tycon='Type', tyvars=('A', 'B')), rhs=(800, 600, 'A', 'B'))
ex20 = tydecl(lhs=simpletype(nargs=2, tycon='Type', tyvars=('A', 'B')), rhs=(None,))
ex21 = tydecl(lhs=simpletype(nargs=2, tycon='Type', tyvars=('A', 'B')), rhs=(('A', 'B'),))
ex22 = tydecl(lhs=simpletype(nargs=0, tycon='Type', tyvars=()), rhs=[('A', 'B'), [('C', 'D'), [('E', ('A', 'B')), None]]])
ex23 = tydecl(lhs=simpletype(nargs=2, tycon='Type', tyvars=('a', 'b')), rhs=[('A', 'B'), [('C', 'D'), [('E', ('A', 'B')), None]]])
ex24 = tydecl(lhs=simpletype(nargs=0, tycon='Type', tyvars=()), rhs=(('A', [('B', 0), [('C', 0), None]]),))
ex25 = tydecl(lhs=simpletype(nargs=0, tycon='Stock', tyvars=()), rhs=[('name', 'string'), [('min', 'int64'), [('max', 'int64'), [('mid', 'int64'), [('volume', 'float'), [('close', 'float'), [('open', 'float'), None]]]]]]])

@skip
def test_all_the_strings():

    parser = DatashapeParser()

    parser.parse('a')
    parser.parse('a, b')
    parser.parse('a, b, c')
    parser.parse('a,      b')
    parser.parse('a,      b  ,     d')
    parser.parse('foo = c')
    parser.parse('a b c = d,e,f')
    parser.parse('   a b c = d, e   ')
    parser.parse('foo a b = c, d')
    parser.parse('foo a b = c,d,e')
    parser.parse('foo a b = c,d,e,   f')
    parser.parse('foo a b = c,   d,   e,   f')
    parser.parse('foo b = c,   d,   e,   f')
    parser.parse('a b c = d, e')
    parser.parse('a b c = bar, foo')
    parser.parse('800, 600, RGBA')
    parser.parse('Pixel A = A')
    parser.parse('Pixel A = A, B')
    parser.parse('Pixel A = 800, 600, A, A')
    parser.parse('Type A B = 800, 600, A, B')
    parser.parse('Type A B = {}')
    parser.parse('Type A B = {A:B}')
    parser.parse('''
    Type = {
        A: B,
        C: D,
        E: (A,B),
    }
    ''')

    parser.parse('''
    Type a b = {
        A:B,
        C:D,
        E:(A,B),
    }
    ''')

    parser.parse('''
    Type = {
        A:({
            B: 0,
            C: 0,
        })
    }
    ''')

    parser.parse('''
    Stock = {
      name   : string,
      min    : int64,
      max    : int64,
      mid    : int64,
      volume : float,
      close  : float,
      open   : float,
    }
    ''')
