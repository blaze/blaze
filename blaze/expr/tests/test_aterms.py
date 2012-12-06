from blaze.expr.paterm import *

def test_simple():
    a = ATerm('a')
    b = ATerm('b')
    annot = ('contig', 'fizzy', 'b')
    annotation = AAnnotation(annot)
    lst = AList(a, b, annotation=annotation)

    assert annotation['type']
    assert not annotation['a']
    assert annotation

    apl = AAppl(ATerm('foo'), [a,b])

    assert matches('*;*', apl)

def test_query_metadata():
    a = ATerm('a')
    b = ATerm('b')
    Add = ATerm('Add')

    annot = ('foo', 'bar')
    annotation = AAnnotation(annot)
    aterm = AAppl(Add, [a,b], annotation=annotation)

    #------------------

    match = matches('Add(x,y);*', aterm)

    match['x'] == a
    match['y'] == b

    #------------------

    match = matches('f(x,y);*', aterm)

    match['f'] == Add
    match['x'] == a
    match['y'] == b

    #------------------

    match = matches(('*;foo'), aterm)
    assert match

    #------------------

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
            return t
        elif isinstance(t, AAppl):
            return All(matcher)(t)

    assert All(matcher)(term3)
    #assert Innermost(matcher)(term)

def test_matching_rewriter():
    a = ATerm('a')
    b = ATerm('b')
    F = ATerm('F')

    term  = AAppl(F, [a,b])

    def matcher(t):
        m = matches('F(x,y);*', t)
        if m:
            return AAppl(F, [m['y'], m['x']])
        else:
            return t

    assert repr(Bottomup(matcher)(term)) == 'F(b,a)'
