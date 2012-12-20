from blaze.expr.uaterm import *

def test_parser_sanity():
    parser = ATermParser()

    parser.parse('f')
    parser.parse('f(x)')
    parser.parse('f(x,y)')
    parser.parse('(x,y)')
    parser.parse('f(x,f(x,y))')
    parser.parse('f(1,2,3)')
    parser.parse('f([1,2,3])')
    parser.parse('(1,2,3)')

    parser.parse('f(x,y){abc, foo}')
    parser.parse('f(x,y){abc, foo, awk}')
    parser.parse('f(x,y{fizzbang})')
    parser.parse('f{x}')
    parser.parse('f{[(a,b),(c,d)]}')

    parser.parse('Pow(<term>,<term>)')

    parser.parse('Mul(Array(),Array())')
    parser.parse('Mul(Array,Array)')
    parser.parse('Add(2,3{dshape("foo, bar, 2")})')
    parser.parse('Add(2{dshape("int"),62764584},3.0{dshape("double"),62764408})')

    parser.parse('[1,2, 3]')
    parser.parse('["foo"]')
    parser.parse('[1, 2, 3.14]')
    parser.parse('[a,b,c,f(b),f(b,1),1.2,1.3]')
    parser.parse('[c,f(b,1),a,b,f(b),1.3,1.2]')
    parser.parse('[[1,2],[1]]')
    parser.parse('[]{[a,a]}')
    parser.parse('f(<int>,<real>,<placeholder>,<str>)')
    parser.parse('f(<list>,[])')
    parser.parse('<appl(1,2)>')
    parser.parse('<term>{[a,b]}')


    parser.matches('x', 'x')
    parser.matches('x', 'y')
    parser.matches('x{foo}', 'x{foo}')

    parser.matches('f(x,y)', 'f(x,y)')
    parser.matches('f(x,g(x,y))', 'f(x,g(x,y))')
    parser.matches('f(<int>,g(x,y))', 'f(1,g(x,y))')
    parser.matches('f(<int>,g(x,y))', 'f(1,g(x,y))')
    parser.matches('f(1,<appl(x,y)>)', 'f(1,g(x,y))')
    parser.matches('f(1,<appl(x,<term>)>)', 'f(1,g(x,3))')

    parser.make('f(<int>)', aint(1))
    parser.make('f(x, y, g(<int>,<int>))', aint(1), aint(2))
    parser.make('<appl(x,y)>', aterm('x', None))
