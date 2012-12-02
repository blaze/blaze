from ndtable.expr.paterm import AAppl, ATerm, AList, AAnnotation
from ndtable.expr.paterm import *

def test_simple():
    a = ATerm('a')
    b = ATerm('b')
    lst =  AList(a,b)
    annot = ('contig', 'fizzy', 'b')
    aterm = AAnnotation(lst, int, annot)

    assert aterm['type']
    assert not aterm['a']
    #assert matches('*;fizzy,b', aterm)
    assert aterm

    apl = AAppl(ATerm('foo'), [a,b])

def test_query_metadata():
    a = ATerm('a')
    b = ATerm('b')
    lst =  AAppl(ATerm('f'), [a,b])
    annot = ('foo', 'bar')
    aterm = AAnnotation(lst, int, annot)

    #------------------

    match = matches(('f(x,y);*'), aterm)

    match['x'] == a
    match['y'] == b

    #------------------

    match = matches(('*;foo'), aterm)
    assert match

    #------------------

    match = matches(('*;fizz'), aterm)
    assert not match

def test_simple_rewrite():
    a = ATerm('a')
    b = ATerm('b')

    term  = AAppl(ATerm('f'), [a,b])
    term2 = AAppl(ATerm('g'), [term, term])
    term3 = AAppl(ATerm('h'), [term2, term2])

    assert repr(Repeat(Fail())(a)) == 'a'
    assert repr(Repeat(Fail())(term)) == 'f(a,b)'

    assert repr(All(lambda x: b)(term)) == 'f(b,b)'
    assert repr(Seq(All(lambda y: a), All(lambda x: b))(term)) == 'f(b,b)'

    assert repr(Topdown(Try(lambda x: a))(term3)) == 'a'
    assert repr(Topdown(Try(All(lambda x: a)))(term3)) == 'h(a,a)'

    assert repr(Bottomup(Try(lambda x: a))(term3)) == 'a'

    def matcher(t):
        if isinstance(t, ATerm):
            return 1
        elif isinstance(t, AAppl):
            return All(matcher)(t)

    assert All(matcher)(term3) #assert Innermost(matcher)(term)
