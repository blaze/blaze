from blaze.datashape.parser import parse

def test_all_the_strings():
    print parse('a')
    print parse('a, b')
    print parse('a, b, c')
    print parse('a,      b')
    print parse('a,      b  ,     d')
    print parse('800, 600, RGBA')
    print parse('type foo = c')
    print parse('type foo    =    c')
    print parse('type a b c = d,e,f')
    print parse('type   a b c = d, e   ')
    print parse('type foo a b = c, d')
    print parse('type foo a b = c,d,e')
    print parse('type foo a b = c,d,e,   f')
    print parse('type foo a b = c,   d,   e,   f')
    print parse('type foo b = c,   d,   e,   f')
    print parse('type a b c = d, e')
    print parse('type a b c = bar, foo')
    print parse('type Pixel A = A')
    print parse('type Pixel A = A, B')
    print parse('type Pixel A = 800, 600, A, A')
    print parse('type Pixel A B = 800, 600, A, B')
    print parse('type Pixel A B = {}')
    print parse('type Pixel A B = {A:B}')
    print parse('''
    type foo = {
        A: B;
        C: D;
        E: (A,B)
    }
    ''')

    print parse('''
    type bar a b = {
        A : B;
        C : D;
        E : (a,b)
    }
    ''')

    parse('type empty = {} ')

    print parse('''
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
