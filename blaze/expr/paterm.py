"""
Python implementation of ATerm. Just a pretty printer for now.

In our slight modification of ATerm we require that the first
annotation of a Term be its datashape type and the following
annotations stand for metadata. For example, an Array.

Type::

    Array(2928121){dshape('2,2,int32')}

Type and Metadata::

    Array(2928121){dshape('2,2,int32'), contigious, corder}

Nevertheless, it is still a strict subset of ATerm so it can be parsed
and manipulated by Stratego rewriters.

Grammar::

    t : bt                 -- basic term
      | bt {ty,m1,...}     -- annotated term

    bt : C                 -- constant
       | C(t1,...,tn)      -- n-ary constructor
       | [t1,...,tn]       -- list
       | "ccc"             -- quoted string
       | int               -- integer
       | real              -- floating point number

Examples::

    >>> AInt(1)
    1
    >>> ATerm('x')
    x
    >>> AAppl(ATerm('Add'), [AInt(1), AInt(2)])
    Add(1,2)
    >>> AAppl(ATerm('Mul'),[AAppl(ATerm('Add'),[AInt(1),AInt(2)]),ATerm('x')])
    Mul(Add(1,2), x)


Pattern Matching::

    >>> matches('foo;*', ATerm('foo'))
    True

    >>> matches('1;*', AInt('1'))
    True

    >>> matches('1;*', AInt('2'))
    True

Value Capturing::

Pattern matching can als also be used to capture free variables
in the matched expressions. The result is also "truthy" so it
will match as a boolean True as well.

    >>> fn = AAppl(ATerm('F'), [ATerm('a')])
    # Upper case variables are bound terms, lower case variables
    # are free

    # Fixed spine
    >>> matches('F(x,y);*', AAppl(ATerm('F'), [ATerm('a'), ATerm('b')]))
    {'y': b, 'x': a}

    # Pattern match includes spine
    >>> matches('f(x,y);*', AAppl(ATerm('F'), [ATerm('a'), ATerm('b')]))
    {'y': b, 'x': a, 'f': F}

Annotation Matching::

>>> at = ATerm('x', annotation=AAnnotation(AString("foo")))
>>> matches('*;foo', at)
True

>>> matches('x;bar', at)


# TODO: recursive pattern matching

"""
import re
from functools import partial
from collections import OrderedDict

sep = re.compile("[\(.*\)]")

#------------------------------------------------------------------------
# Terms
#------------------------------------------------------------------------

class ATermBase(object):
    "Base for aterms"

    # Specifies child attributes
    _fields = []

    def __init__(self, annotation=None):
        self.annotation = annotation

    @property
    def metastr(self):
        if self.annotation:
            return str(self.annotation)
        else:
            return ''

class ATerm(ATermBase):

    def __init__(self, label, **kwargs):
        super(ATerm, self).__init__(**kwargs)
        self.label = label

    def __str__(self):
        return str(self.label) + self.metastr

    def _matches(self, query):
        return self.label == query

    def __repr__(self):
        return str(self)

class AAppl(ATermBase):

    _fields = ['spine', 'args']

    def __init__(self, spine, args, **kwargs):
        super(AAppl, self).__init__(**kwargs)
        self.spine = spine
        self.args = args

    def _matches(self, query):
        if query == '*':
            return True

        spine, args, _ = sep.split(query)
        args = args.split(',')

        if len(self.args) != len(args):
            return False

        # success
        if spine.islower() or self.spine.label == spine:
            _vars = {}
            argm = [b.islower() or a._matches(b) for a,b in zip(self.args, args)]

            if spine.islower():
                _vars[spine] = self.spine

            for i, arg in enumerate(args):
                if argm[i]:
                    _vars[arg] = self.args[i]

            return _vars

        else:
            return False

    def __str__(self):
        return str(self.spine) + cat(self.args, '(', ')') + self.metastr

    def __repr__(self):
        return str(self)

class AAnnotation(ATermBase):

    def __init__(self, ty=None, annotations=None):
        self.ty = ty or None

        if annotations is not None:
            # Convert annotations to aterms
            annotations = list(annotations)
            for i, annotation in enumerate(annotations):
                if not isinstance(annotation, ATermBase):
                    annotations[i] = ATerm(annotation)

        self.meta = annotations or []

    @property
    def annotations(self):
        terms = map(ATerm, (self.ty,) + tuple(self.meta))
        return cat(terms, '{', '}')

    def __contains__(self, key):
        if key == 'type':
            return True
        else:
            return key in self.meta

    def __getitem__(self, key):
        if key == 'type':
            return self.ty
        else:
            return key in self.meta

    def _matches(self, value, meta):
        if value == '*':
            vmatch = True
        else:
            vmatch = self.bt._matches(value)

        if meta == ['*']:
            mmatch = True
        else:
            mmatch = all(a in self for a in meta)

        if vmatch and mmatch:
            return vmatch
        else:
            return False

    def __str__(self):
        return self.annotations

    def __repr__(self):
        return str(self)

class AString(ATermBase):
    def __init__(self, s, **kwargs):
        super(AString, self).__init__(**kwargs)
        self.s = s

    def __str__(self):
        return repr(self.s) + self.metastr

    def __repr__(self):
        return str(self)

class AInt(ATermBase):
    def __init__(self, n, **kwargs):
        super(AInt, self).__init__(**kwargs)
        self.n = n

    def _matches(self, value):
        if value.islower():
            return True
        else:
            return self.n == int(value)

    def __str__(self):
        return str(self.n) + self.metastr

    def __repr__(self):
        return str(self)

class AFloat(ATermBase):
    def __init__(self, n, **kwargs):
        super(AFloat, self).__init__(**kwargs)
        self.n = n

    def __str__(self):
        return str(self.n) + self.metastr

    def __repr__(self):
        return str(self)

class AList(ATermBase):
    def __init__(self, *elts, **kwargs):
        super(AList, self).__init__(**kwargs)
        self.elts = elts

    def __str__(self):
        return cat(self.elts, '[', ']') + self.metastr

    def __repr__(self):
        return str(self)


#------------------------------------------------------------------------
# Strategic Rewrite Combinators
#------------------------------------------------------------------------

Id = lambda s: s

def Fail():
    raise STFail()

class STFail(Exception):
    pass

compose = lambda f, g: lambda x: f(g(x))

class Fail(object):

    def __init__(self):
        pass

    def __call__(self, o):
        raise STFail()

class Choice(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        assert left and right, 'Must provide two arguments to Choice'

    def __call__(self, s):
        try:
            return self.left(s)
        except STFail:
            return self.right(s)

class Ternary(object):
    def __init__(self, s1, s2, s3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def __call__(self, o):
        try:
            val = self.s1(o)
        except STFail:
            return self.s2(val)
        else:
            return self.s3(val)

class Fwd(object):

    def __init__(self):
        self.p = None

    def define(self, p):
        self.p = p

    def __call__(self, s):
        if self.p:
            return self.p(s)
        else:
            raise NotImplementedError('Forward declaration, not declared')

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
        if isinstance(o, AAppl):
            return AAppl(o.spine, map(self.s, o.args))
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

#------------------------------------------------------------------------
# Pattern Matching
#------------------------------------------------------------------------

def matches(pattern, term):
    """
    Collapse terms with as-patterns.
    """

    value, meta = pattern.replace(' ','').split(';')
    meta = meta.split(',')

    if isinstance(term, AAnnotation):
        return term._matches(value, meta)
    else:
        return term._matches(value)

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def cat(terms, l, r):
    """ Concatenate str representations with commas and left and right
    delimiters. """
    return l + ','.join(map(str, terms)) + r
