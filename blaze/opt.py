"""
"""

import pyrewrite

# We'd like to be just be able to quote these rules inline... ala
# Mython.

#------------------------------------------------------------------------
# Constant Folding
#------------------------------------------------------------------------

const = """

cfold: Add(Const(n), Const(m))   -> Const(k) where { k = n+m }
cfold: Mul(Const(n), Const(m))   -> Const(k) where { k = n*m }
cfold: Sub(Const(n), Const(m))   -> Const(k) where { k = n-m }
cfold: Div(Const(n), Const(m))   -> Const(k) where { k = n/m }
cfold: Mod(Const(n), Const(m))   -> Const(k) where { k = n%m }
cfold: Power(Const(n), Const(m)) -> Const(k) where { k = n**m }

const-fold = repeat(topdown(cfold) <+ id)

"""

const_mod = pyrewrite.module(const)

def constant_fold(expr):
    return const_mod.rewrite(expr)

#------------------------------------------------------------------------
# Conditional Folding
#------------------------------------------------------------------------

cond = """

EvalIf :
    IfElse(False(), e1, e2) -> e2

EvalIf :
    IfElse(True(), e1, e2) -> e1

PropIf :
    IfElse(B,@F(X),@F(Y)) -> F(IfElse(B,X,Y))

cond-fold = repeat(topdown(EvalIf) <+ id)

"""

cond_mod = pyrewrite.module(cond)

def conditional_fold(expr):
    return cond_mod.rewrite(expr)
