from blaze import dshape

from blaze.expr.nodes import Node
from blaze.expr.viz import dump
from blaze.table import NDArray, NDArray

from blaze.datashape.coretypes import float64, dynamic
from blaze.expr.graph import IntNode, FloatNode, App, StringNode

from unittest import skip

# Print out graphviz to the screen
DEBUG = False

def test_walk():
    e = Node([])
    d = Node([])
    b = Node([d])
    c = Node([e])
    a = Node([b,c])

    assert len([n for n in a]) == 4

    if DEBUG:
        dump(a, filename='walk', tree=True)

@skip
def test_dynamic_arguments():
    a = NDArray([])
    b = NDArray([a])

    children = b.children
    assert len(children) == 1

def test_binary_ops():
    a = IntNode(1)
    b = IntNode(2)

    x = a+b

    if DEBUG:
        dump(x, filename='binary')

def test_binary_mixed():
    a = IntNode(1)

    x = a+2

    if DEBUG:
        dump(x, filename='binarymixed')

def test_unary_ops():
    a = IntNode(1)

    x = abs(a)

    if DEBUG:
        dump(x, filename='unary')

def test_indexing():
    a = NDArray([])

    x = a[0]

    if DEBUG:
        dump(x, filename='indexer')

def test_slice():
    a = NDArray([])

    x = a[0:1]

    if DEBUG:
        dump(x, filename='slice')

def test_scalars():
    a = IntNode(1)
    b = IntNode(1)
    c = IntNode(2)

    x = abs((a + b + c + 3) * 4)

    if DEBUG:
        dump(x, filename='scalars')
