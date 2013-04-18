from rewrite import parse

def test_parser_sanity():
    a0 = parse('f')
    a1 = parse('f(x)')
    a2 = parse('f(x,y)')
    a3 = parse('(x,y)')
    a4 = parse('f(x,f(x,y))')
    a5 = parse('f(1,2,3)')
    a6 = parse('f([1,2,3])')
    a7 = parse('(1,2,3)')

    assert repr(a0) == 'f'
    assert repr(a1) == 'f(x)'
    assert repr(a2) == 'f(x, y)'
    assert repr(a3) == '(x, y)'
    assert repr(a4) == 'f(x, f(x, y))'
    assert repr(a5) == 'f(1, 2, 3)'
    assert repr(a6) == 'f([1, 2, 3])'
    assert repr(a7) == '(1, 2, 3)'

    # test edge case
    assert repr(parse('[]')) == '[]'
    assert repr(parse('()')) == '()'

    parse('f(x,y){abc, foo}')
    parse('f(x,y){abc, foo, awk}')
    parse('f(x,y{fizzbang})')
    parse('f{x}')
    parse('f{[(a,b),(c,d)]}')

    parse('[1,2, 3]')
    parse('["foo"]')
    parse('[1, 2, 3.14]')
    parse('[a,b,c,f(b),f(b,1),1.2,1.3]')
    parse('[c,f(b,1),a,b,f(b),1.3,1.2]')
    parse('[[1,2],[1]]')
    parse('[]{[a,a]}')
    parse('f(<int>,<real>,<placeholder>,<str>)')
    parse('f(<list>,[])')
    parse('<appl(1,2)>')
    parse('<term>{[a,b]}')

    parse('Pow(<term>,<term>)')
    parse('Mul(Array(),Array())')
    parse('Mul(Array,Array)')
    parse('Add(2,3{dshape("foo, bar, 2")})')
    parse('Add(2{dshape("int")},3.0{dshape("double")})')

def test_roundtrip():
    a0 = parse(repr(parse('f')))
    a1 = parse(repr(parse('f(x)')))
    a2 = parse(repr(parse('f(x, y)')))
    a3 = parse(repr(parse('(x, y)')))
    a4 = parse(repr(parse('f(x, f(x, y))')))
    a5 = parse(repr(parse('f(1, 2, 3)')))
    a6 = parse(repr(parse('f([1, 2, 3])')))
    a7 = parse(repr(parse('(1, 2, 3)')))

    assert repr(a0) == 'f'
    assert repr(a1) == 'f(x)'
    assert repr(a2) == 'f(x, y)'
    assert repr(a3) == '(x, y)'
    assert repr(a4) == 'f(x, f(x, y))'
    assert repr(a5) == 'f(1, 2, 3)'
    assert repr(a6) == 'f([1, 2, 3])'
    assert repr(a7) == '(1, 2, 3)'
