import blaze.module.typing
import blaze.module.nodes as N
from blaze.module.parser import mread

def test_resolution1():

    example = \
    """
    module Test1 {

        trait Foo[A t]:
            fun lt :: (A t, A t) -> (A bool)

        impl Foo[Array t]:
            fun lt = pass
    }

    """

    mod = mread(example)
    sig = mod.bound_ns['Array']['lt']


def test_resolution2():

    example = \
    """
    module Test1 {

        trait Foo[A t]:
            fun lt :: t , t -> t

        impl Foo[Array x]:
            fun lt = pass
    }

    """

    mod = mread(example)
    sig = mod.bound_ns['Array']['lt']

def test_typeset():

    example = \
    """
    module Test1 {

        typeset simple = int | float | bool

        trait Foo[A t]:
            fun lt :: t , t -> t

        impl Foo[Array x] for (x in simple):
            fun lt = pass
    }

    """

    mod = mread(example)
    sig = mod.bound_ns['Array']['lt']
