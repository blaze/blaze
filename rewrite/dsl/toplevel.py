from functools import partial

from rewrite.matching import free, freev, fold, unfold, hylo, NoMatch
import rewrite.astnodes as ast

from parse import dslparse
import combinators as comb

def nameof(o):
    if isinstance(o, RuleBlock):
        return o.label
    if isinstance(o, Strategy):
        return o.label
    else:
        return o.__class__.__name__

combinators = {
    'fail'      : comb.fail,
    'id'        : comb.Id,
    'repeat'    : comb.Repeat,
    'all'       : comb.All,
    'some'      : comb.Some,
    '<+'        : comb.Choice,
    ';'         : comb.Seq,
    'try'       : comb.Try,
    'topdown'   : comb.Topdown,
    'bottomup'  : comb.Bottomup,
    'innermost' : comb.Innermost,
    'debug'     : comb.Debug,
}

#------------------------------------------------------------------------
# Strategies
#------------------------------------------------------------------------

class Strategy(object):

    def __init__(self, combinator, expr, label):
        self.subrules = [a for a in expr]

        try:
            self.combinator = combinator(*self.subrules)
        except TypeError:
            raise TypeError, 'Wrong number of arguments to combinator: %s'\
                % str(combinator)

        self.label = label or repr(self)

    def __call__(self, o):
        return self.combinator(o)

    rewrite = __call__

    def __repr__(self):
        return '%s(%s)' % (
            nameof(self.combinator),
            ','.join(map(nameof, self.subrules))
        )

class Rule(object):
    def __init__(self, lpat, rpat, left, right, rr):
        self.lpat = lpat
        self.rpat = rpat
        self.left = left
        self.right = right
        self.rr = rr

    def rewrite(self, subject):
        return self.rr(subject)

    __call__ = rewrite

    def __repr__(self):
        return '%r => %r ::\n\t %r -> %r' % \
            (self.left, self.right, self.lpat, self.rpat)

class RuleBlock(object):
    def __init__(self, rules=None, label=None):
        self.rules = rules or []
        self.label = label

    def add(self, rule):
        self.rules.append(rule)

    def rewrite(self, pattern):
        for rule in self.rules:
            try:
                return rule.rewrite(pattern)
            except NoMatch:
                continue
        raise NoMatch()

    def __call__(self, pattern):
        return self.rewrite(pattern)

    def __repr__(self):
        out = '[\n'
        for rule in self.rules:
            out += ' '*4 + repr(rule) + '\n'
        out += ']\n'
        return out

#------------------------------------------------------------------------
# Buld Automata
#------------------------------------------------------------------------

def build_strategy(label, env, comb, args):
    env = env.copy()

    self = object() # forward declaration since rules can be self-recursive
    comb = combinators[comb]

    env.update(combinators)

    sargs = []

    for arg in args:
        # composition of combinators
        if isinstance(arg, tuple):
            subcomb, subargs = arg
            sargs.append(build_strategy(None, env, subcomb, subargs))

        if isinstance(arg, list):
            for iarg in arg:
                # look up the corresponding rewrite rule or
                # rewrite block and pass the rewrite hook to the
                # strategy combinator
                rr = env[iarg.term]
                sargs.append(rr)

    return Strategy(comb, sargs, label)

def build_rule(l, r, cons=None):
    lpat = []
    rpat = []
    sym = set()
    cons = cons or {}

    for pat, bind, ty in free(l):
        if bind in sym:
            # TODO unify ty
            lpat.append(bind)
        else:
            lpat.append(bind)
            sym.add(bind)

    for pat, bind, ty in free(r):
        if bind in sym:
            rpat.append(bind)
        else:
            raise Exception('Unbound variable: %s' % pat)

    left  = freev(l)
    right = freev(r)

    ana  = partial(unfold, lpat, left)
    cata = partial(fold, rpat, right)

    rr = partial(hylo, ana, cata)
    return Rule(lpat, rpat, left, right, rr)

#------------------------------------------------------------------------
# Module Constructions
#------------------------------------------------------------------------

def module(s, sorts=None, cons=None, _env=None):
    defs = dslparse(s)

    if _env:
        env = _env.copy()
    else:
        env = {}

    for df in defs:

        if isinstance(df, ast.RuleNode):

            label, l, r = df
            rr = build_rule(l, r, cons)

            if label in env:
                env[label].add(rr)
            else:
                env[label] = RuleBlock([rr], label=label)

        elif isinstance(df, ast.StrategyNode):
            label, comb, args = df

            if label in env:
                raise Exception, "Strategy definition '%s' already defined" % label

            st = build_strategy(label, env, comb, args)
            env[label] = st

        else:
            raise NotImplementedError

    return env
