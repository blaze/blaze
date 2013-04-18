from terms import * # TODO: tsk, tsk
import astnodes as ast

placeholders = {
    'appl'        : aappl,
    'str'         : astr,
    'int'         : aint,
    'real'        : areal,
    'term'        : (aterm, aappl, astr, aint, areal),
    'placeholder' : aplaceholder,
    'list'        : alist
}

class NoMatch(Exception):
    pass

#------------------------------------------------------------------------
# Traversal
#------------------------------------------------------------------------

def init(xs):
    for x in xs:
        return x

def aterm_zip(a, b):
    if isinstance(a, (aint, areal, astr)) and isinstance(b, (aint, areal, astr)):
        yield a.val == b.val, None

    elif isinstance(a, aterm) and isinstance(b, aterm):
        yield a.term == b.term, None
        yield a.annotation == b.annotation, None

    elif isinstance(a, aappl) and isinstance(b, aappl):
        if len(a.args) == len(b.args):
            yield a.spine == b.spine, None
            for ai, bi in zip(a.args, b.args):
                for aj in aterm_zip(ai,bi):
                    yield aj
        else:
            yield False, None

    elif isinstance(a, atupl) and isinstance(b, atupl):
        if len(a.args) == len(b.args):
            for ai, bi in zip(a.args, b.args):
                for aj in aterm_zip(ai,bi):
                    yield aj
        else:
            yield False, None

    elif isinstance(a, aplaceholder):
        # <appl(...)>
        if a.args:
            if isinstance(b, aappl):
                yield True, b.spine
                for ai, bi in zip(a.args, b.args):
                    for a in aterm_zip(ai,bi):
                        yield a
            else:
                yield False, None
        # <term>
        else:
            yield isinstance(b, placeholders[a.type]), b
    else:
        yield False, None


# left-to-right substitution
def aterm_splice(a, elts):

    if isinstance(a, aterm):
        yield a

    elif isinstance(a, (aint, areal, astr)):
        yield a

    elif isinstance(a, aappl):
        yield aappl(a.spine, [init(aterm_splice(ai,elts)) for ai in a.args])

    elif isinstance(a, atupl):
        yield atupl([init(aterm_splice(ai,elts)) for ai in a.args])

    elif isinstance(a, aplaceholder):
        # <appl(...)>
        if a.args:
            spine = elts.pop(0)
            yield aappl(spine, [init(aterm_splice(ai,elts)) for ai in a.args])
        # <term>
        else:
            yield elts.pop(0)
    else:
        raise NotImplementedError

#------------------------------------------------------------------------
# Rewriting
#------------------------------------------------------------------------

# TODO: warn on nested as pattern
def free(a):
    if isinstance(a, (aint, areal, astr)):
        pass

    elif isinstance(a, aappl):
        for ai in a.args:
            for aj in free(ai):
                yield aj

    elif isinstance(a, aterm):
        yield (a, a.term, aterm)

    elif isinstance(a, (alist,atupl)):
        for ai in a.args:
            for aj in free(ai):
                yield aj

    # ----

    elif isinstance(a, ast.AsNode):
        if a.tag is not None:
            yield (a.pattern, a.tag, type(a.pattern))
        else:
            if isinstance(a.pattern, aappl):
                # detached
                head = a.pattern.spine
                yield (head, head.term, aterm)
                for ai in a.pattern.args:
                    for aj in free(ai):
                        yield aj

    else:
        raise NotImplementedError


def freev(a):
    if isinstance(a, (aint, areal, astr)):
        return a

    elif isinstance(a, aappl):
        return aappl(a.spine, [freev(ai) for ai in a.args])

    elif isinstance(a, alist):
        return alist([freev(ai) for ai in a.args])

    elif isinstance(a, atupl):
        return atupl([freev(ai) for ai in a.args])

    elif isinstance(a, aterm):
        return aplaceholder('term', None)

    # ----

    elif isinstance(a, ast.AsNode):
        # TODO: need to define traversal
        if a.tag is not None:
            return aplaceholder('term', None)
        else:
            return aplaceholder('appl', [freev(ai) for ai in a.pattern.args])

    else:
        raise NotImplementedError

#------------------------------------------------------------------------
# Rewriting
#------------------------------------------------------------------------

def unfold(lpat, p, s):
    bindings = {}
    li = iter(lpat)

    for matches, capture in aterm_zip(p,s):
        if not matches:
            raise NoMatch()
        elif matches and capture:
            bind = next(li)
            # check non-linear or nested patterns
            if bind in bindings:
                if capture == bindings[bind]:
                    yield (bind, capture)
                else:
                    raise NoMatch()
            # flat patterns
            else:
                bindings[bind] = capture
                yield (bind, capture)

def fold(rpat, subst, cap):
    stack = dict(cap)

    vals = [stack[s] for s in rpat]

    return init(aterm_splice(subst,vals))

def hylo(ana, cata, s):
    return cata(ana(s))
