#------------------------------------------------------------------------
# Terms
#------------------------------------------------------------------------

class ATerm(object):

    def __init__(self, term, annotation=None):
        self.term = term
        self.annotation = annotation

    def __str__(self):
        if self.annotation is not None:
            return str(self.term) + arepr([self.annotation], '{', '}')
        else:
            return str(self.term)

    def __eq__(self, other):
        if isinstance(other, ATerm):
            return self.term == other.term
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

    def __show__(self):
        pass

class AAppl(object):

    def __init__(self, spine, args):
        assert isinstance(spine, ATerm)
        self.spine = spine
        self.args = args

    def __eq__(self, other):
        if isinstance(other, AAppl):
            return self.spine == other.spine and self.args == other.args
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return str(self.spine) + arepr(self.args, '(', ')')

    def __repr__(self):
        return str(self)

    def __show__(self):
        pass

class AString(object):
    def __init__(self, val):
        assert isinstance(val, str)
        self.val = val

    def __str__(self):
        return '"%s"' % (self.val)

    def __repr__(self):
        return str(self)

class AInt(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, AInt):
            return self.val == other.val
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

class AReal(object):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if isinstance(other, AReal):
            return self.val == other.val
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

class AList(object):
    def __init__(self, args):
        assert isinstance(args, list)
        self.args = args or []

    def __str__(self):
        return arepr(self.args, '[', ']')

    def __eq__(self, other):
        if isinstance(other, AList):
            return self.args == other.args
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

class ATuple(object):
    def __init__(self, args):
        assert isinstance(args, list)
        self.args = args or []

    def __eq__(self, other):
        if isinstance(other, ATuple):
            return self.args == other.args
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return arepr(self.args, '(', ')')

    def __repr__(self):
        return str(self)

class APlaceholder(object):

    def __init__(self, type, args):
        self.type = type
        self.args = args

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

# Goal is to only define the protocol that these need to conform
# to and then allow pluggable backends.

#    - ATerm
#    - Python AST
#    - SymPy

aterm = ATerm
aappl = AAppl
aint  = AInt
astr  = AString
areal = AReal
atupl = ATuple
alist = AList
aplaceholder = APlaceholder
