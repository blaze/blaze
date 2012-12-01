from ndtable.expr.paterm import AAppl, ATerm, AList, AAnnotation
from ndtable.expr.visitor import MroTransformer

def test_simple():
    a = ATerm('a')
    b = ATerm('b')
    lst =  AList(a,b)
    annot = ('contig', 'fizzy', 'b')
    aterm = AAnnotation(lst, int, annot)

    assert aterm['type']
    assert not aterm['a']
    assert aterm.matches('*;fizzy,b')
    assert aterm

    apl = AAppl(ATerm('foo'), [a,b])

def test_query_metadata():
    a = ATerm('a')
    b = ATerm('b')
    lst =  AList(a,b)
    annot = ('contig', 'fizzy', 'b')
    aterm = AAnnotation(lst, int, annot)

    assert aterm.matches('*;fizzy,b')
