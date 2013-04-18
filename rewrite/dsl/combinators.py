from functools import wraps
import rewrite.terms as terms

#------------------------------------------------------------------------
# Exceptions
#------------------------------------------------------------------------

class STFail(Exception):
    pass

#------------------------------------------------------------------------
# Rewrite Combinators
#------------------------------------------------------------------------

Id = lambda s: s
compose = lambda f, g: lambda x: f(g(x))

def fix(f):
    @wraps(f)
    def Yf(*args):
        return inner(*args)
    inner = f(Yf)
    return Yf

def fail():
    raise STFail()

class Choice(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        assert left and right, 'Must provide two arguments to Choice'

    def __call__(self, t):
        try:
            return self.left(t)
        except STFail:
            return self.right(t)

class Debug(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, t):
        res = self.s(t)
        print res
        return res

class Ternary(object):
    def __init__(self, s1, s2, s3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def __call__(self, t):
        try:
            val = self.s1(t)
        except STFail:
            return self.s2(val)
        else:
            return self.s3(val)

class Repeat(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, s):
        val = s
        while True:
            try:
                val = self.p(val)
            except STFail:
                break
        return val

class All(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, o):
        if isinstance(o, terms.AAppl):
            return terms.AAppl(o.spine, map(self.s, o.args))
        else:
            return o

class Some(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, o):
        if isinstance(o, AAppl):
            largs = []
            for a in o.args:
                try:
                    largs.append(self.s(a))
                except STFail:
                    largs.append(a)
            return AAppl(o.spine, largs)
        else:
            raise STFail()

class Seq(object):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, o):
        return self.s2(self.s1(o))

class Try(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, o):
        try:
            return self.s(o)
        except STFail:
            return o
        except Exception:
            return o

class Topdown(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, o):
        val = self.s(o)
        return All(self)(val)

class Bottomup(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, o):
        val = All(self)(o)
        return self.s(val)

class Innermost(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, o):
        return Bottomup(Try(Seq(self.s, self)))(o)

class SeqL(object):
    def __init__(self, *sx):
        self.lx = reduce(compose,sx)

    def __call__(self, o):
        return self.lx(o)

class ChoiceL(object):
    def __init__(self, *sx):
        self.sx = reduce(compose,sx)

    def __call__(self, o):
        return self.sx(o)
