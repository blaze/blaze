# Unification Rules
# -----------------
from blaze import NDArray, dshape
from blaze.compile import compile

def test_simple_unify():
    A = NDArray([0], dshape('s, t, int'))
    B = NDArray([0], dshape('u, v, int'))

    C = NDArray([0], dshape('w, x, int'))
    D = NDArray([0], dshape('y, z, int'))

    # ==============
    g = (A*B+C*D)**2
    # ==============

    out = compile(g)
    print out

    # Operator Constraints
    #

    #                A : (s, t)
    #                B : (u, v)
    #                C : (w, x)
    #                D : (y, z)
    #
    #               AB : (a, b)
    #               CD : (c, d)
    #          AB + CD : (e, f)
    #     (AB + CD)**2 : (g, h)

    # Constraint Generation
    # ---------------------

    # t = u, a = s, b = v   in AB
    # x = y, c = w, d = z   in CD
    # a = c = e, b = d = f  in AB + CD
    # e = f = g = h         in (AB + CD)**2

    # Substitution
    # -------------

    # a = b = c = d = e = f = g = h = s = v = w = z
    # t = u
    # x = y

    # Constraint Solution
    # -------------------

    # A : a -> t
    # B : t -> a
    # C : a -> x
    # D : x -> a
