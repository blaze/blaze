import typing
import nodes as N
from proxy import Proxy

from parser import mopen

from operator import add

from nose.tools import assert_raises

from unittest import skip

#------------------------------------------------------------------------

def test_signatures():
   a = mopen('blaze.mod')
   print "Module".center(80, '=')
   print a.show()

#------------------------------------------------------------------------

def test_adhoc():
    a = mopen('blaze.mod')

    ty = ('Array', [('t', 'int')])
    over = a.resolve_adhoc('add', ty)

def test_ford():
    ty = ('Array', [('t', 'int')])
    p = Proxy(ty)

    q = p > p

#------------------------------------------------------------------------

#def test_degen():
#    a = mopen('blaze.mod')
#
#    # need to resolve in the presense of a function type
#    over = a.resolve_adhoc('map', ty)

#------------------------------------------------------------------------

def test_failure():
    pt = ('Array', [('t', 'int')])
    a = Proxy(pt)

    with assert_raises(TypeError):
        not a

#------------------------------------------------------------------------

def test_operations():
    print "Objects".center(80, '=')

    ty = ('Array', [('t', 'int')])

    p = Proxy(ty)
    print p

    print "Tests".center(80, '=')
    assert p.typeof == ty

    b = p.add(2.0)
    c = b.add(2.0)
    d = c.add(2.0)

    print d
    print b.sqrt()
    print p > p
    print (p+p)+p

    print reduce(add, [p,p,p]*50)

    f = p > p
    f = p - p
    print f

if __name__ == '__main__':
    test_signatures()
    test_adhoc()
    test_operations()
