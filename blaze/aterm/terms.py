#------------------------------------------------------------------------
# Terms
#------------------------------------------------------------------------

class ATerm(object):

    def __init__(self, term, annotation=None):
        self.term = term
        self.annotation = annotation # This should be a dict !

    def __str__(self):
        if self.annotation is not None:
            return str(self.term) + arepr([self.annotation], '{', '}')
        else:
            return str(self.term)

    def __eq__(self, other):
        if isinstance(other, ATerm):
            return self.term == other.term
        else:
            raise ValueError()

    def __ne__(self, other):
        if isinstance(other, ATerm):
            return self.term != other.term
        else:
            raise ValueError()

    def __repr__(self):
        return str(self)

class AAppl(object):

    def __init__(self, spine, args, annotation=None):
        assert isinstance(spine, ATerm)
        self.spine = spine
        self.args = args
        self.annotation = annotation

    def __str__(self):
        return str(self.spine) + arepr(self.args, '(', ')')

    def __repr__(self):
        return str(self)

class AString(object):
    def __init__(self, val, annotation=None):
        assert isinstance(val, str)
        self.val = val
        self.annotation = annotation

    def __str__(self):
        return '"%s"' % (self.val)

    def __repr__(self):
        return str(self)

class AInt(object):
    def __init__(self, val, annotation=None):
        self.val = val
        self.annotation = annotation

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, AInt):
            return self.val == other.val
        else:
            raise ValueError()

    def __ne__(self, other):
        if isinstance(other, AInt):
            return self.val != other.val
        else:
            raise ValueError()

    def __repr__(self):
        return str(self)

class AReal(object):
    def __init__(self, val, annotation=None):
        self.val = val
        self.annotation = annotation

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self)

class AList(object):
    def __init__(self, args, annotation=None):
        assert isinstance(args, list)
        self.args = args or []
        self.annotation = annotation

    def __str__(self):
        return arepr(self.args, '[', ']')

    def __repr__(self):
        return str(self)

class ATuple(object):
    def __init__(self, args, annotation=None):
        assert isinstance(args, list)
        self.args = args or []
        self.annotation = annotation

    def __str__(self):
        return arepr(self.args, '(', ')')

    def __repr__(self):
        return str(self)

class APlaceholder(object):

    def __init__(self, type, args, annotation=None):
        self.type = type
        self.args = args
        self.annotation = annotation

    def __str__(self):
        if self.args is not None:
            return '<%s(%r)>' % (self.type, self.args)
        else:
            return arepr([self.type], '<', '>')

    def __repr__(self):
        return str(self)


#------------------------------------------------------------------------
# Pretty Printing
#------------------------------------------------------------------------

def arepr(terms, l, r):
    """ Concatenate str representations with commas and left and right
    characters. """
    return l + ', '.join(map(str, terms)) + r

#------------------------------------------------------------------------
# Compatability
#------------------------------------------------------------------------

aterm = ATerm
aappl = AAppl
aint  = AInt
astr  = AString
areal = AReal
atupl = ATuple
alist = AList
aplaceholder = APlaceholder

#------------------------------------------------------------------------
# Access
#------------------------------------------------------------------------

terms = {
    'aterm'        : aterm,
    'aappl'        : aappl,
    'aint'         : aint,
    'astr'         : astr,
    'areal'        : areal,
    'atupl'        : atupl,
    'alist'        : alist,
    'aplaceholder' : aplaceholder,
}

__all__ = [
    'aterm',
    'aappl',
    'aint',
    'astr',
    'areal',
    'atupl',
    'alist',
    'aplaceholder'
]
