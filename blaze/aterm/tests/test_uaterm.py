from blaze.aterm.uaterm import *

def test_parser_sanity():

    parse('f')
    parse('f(x)')
    parse('f(x,y)')
    parse('(x,y)')
    parse('f(x,f(x,y))')
    parse('f(1,2,3)')
    parse('f([1,2,3])')
    parse('(1,2,3)')

    parse('f(x,y){abc, foo}')
    parse('f(x,y){abc, foo, awk}')
    parse('f(x,y{fizzbang})')
    parse('f{x}')
    parse('f{[(a,b),(c,d)]}')

    parse('Pow(<term>,<term>)')

    parse('Mul(Array(),Array())')
    parse('Mul(Array,Array)')
    parse('Add(2,3{dshape("foo, bar, 2")})')
    parse('Add(2{dshape("int"),62764584},3.0{dshape("double"),62764408})')

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


    match('x', 'x')
    match('x', 'y')
    match('x{foo}', 'x{foo}')

    match('f(x,y)', 'f(x,y)')
    match('f(x,g(x,y))', 'f(x,g(x,y))')
    match('f(<int>,g(x,y))', 'f(1,g(x,y))')
    match('f(<int>,g(x,y))', 'f(1,g(x,y))')
    match('f(1,<appl(x,y)>)', 'f(1,g(x,y))')
    match('f(1,<appl(x,<term>)>)', 'f(1,g(x,3))')

    build('f(<int>)', aint(1))
    build('f(x, y, g(<int>,<int>))', aint(1), aint(2))
    build('<appl(x,y)>', aterm('x', None))
