from blaze.datashape.parser import parse

from unittest import skip

def test_all_the_strings():
    parse('a')
    parse('a, b')
    parse('a, b, c')
    parse('a,      b')
    parse('a,      b  ,     d')
    parse('800, 600, RGBA')
    parse('type foo = c')
    parse('type foo    =    c')
    parse('type a b c = d,e,f')
    parse('type   a b c = d, e   ')
    parse('type foo a b = c, d')
    parse('type foo a b = c,d,e')
    parse('type foo a b = c,d,e,   f')
    parse('type foo a b = c,   d,   e,   f')
    parse('type foo b = c,   d,   e,   f')
    parse('type a b c = d, e')
    parse('type a b c = bar, foo')
    parse('type bar A = A')
    parse('type bar A = A, B')
    parse('type bar A = 800, 600, A, A')
    parse('type bar A B = 800, 600, A, B')
    parse('type bar A B = {}')
    parse('type bar A B = {A:B}')

    parse('type baz = F(A,B)')
    parse('type baz = F(A,F(A))')

    parse('''
    type foo = {
        A: B;
        C: D;
        E: (A,B)
    }
    ''')

    parse('''
    type bar a b = {
        A : B;
        C : D;
        E : (a,b)
    }
    ''')

    parse('type empty = {} ')

    parse('''
    type Stock = {
      name   : string;
      min    : int64;
      max    : int64;
      mid    : int64;
      volume : float;
      close  : float;
      open   : float
    }
    ''')

def test_trailing_semi():
    a = parse('''
    type a = {
        a: int;
        b: float;
        c: (int,int)
    }
    ''')

    b = parse('''
    type a = {
        a: int;
        b: float;
        c: (int,int);
    }
    ''')

    assert a == b

def test_multiline():
    a = parse('''

    type f a = b
    type g a = b

    type a = {
        a: int;
        b: float;
        c: (int,int);
    }

    ''')

def test_inline():
    a = parse('''
    type Point = {
        x : int;
        y : int
    }

    type Space = {
        a: Point;
        b: Point
    }

    ''')

    a = parse('''
    type Person = {
        name   : string;
        age    : int;
        height : int;
        weight : int
    }

    type RGBA = {
        r: int32;
        g: int32;
        b: int32;
        a: int8
    }
    ''')

def test_nested():
    a = parse('''
    type Space = {
        a: { x: int; y: int };
        b: { x: int; y: int }
    }
    ''')

def test_parameterized():
    a = parse('''
    type T x y = {
        a: x;
        b: y
    }
    ''')

def test_option():
    a = parse(''' Option(int32) ''')

def test_union():
    a = parse(''' Union(a,b,c,d) ''')

def test_either1():
    a = parse(''' Either(int32, float32) ''')

def test_either2():
    a = parse(''' Either({x: int}, {y: float}) ''')

@skip
def test_either3():
    a = parse(''' Either( (2, 2, T), (T, 2, int) ) ''')

def test_stress():
    parse('type big = F(F(F(F(F(F(F(F(F(F(F(F(F(F(F(F(x))))))))))))))))')
    parse('type big = {x:{x:{x:{x:{x:{x:{x:{x:{x:{x:int32}}}}}}}}}}')

    # whitespace insensitivity
    parse('''type big = {
            x:{         x
                :{
                        x:
            {x:{x:
                {x:{x:{  x:{x
        :{x
        :int32
            }}}
                }}    }}}}}
    ''')
