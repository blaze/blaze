from blaze import manifest, delayed, ones
from blaze.rts import execute
from blaze.compile import compile, explain

A = ones('100, 100, int32', eclass=delayed)
B = ones('100, 100, int32', eclass=delayed)

expr = (A + B) * B

# Normally we just use expr.eval() but to demonstrate the compile
# pipeline...

plan = compile(expr)

print plan
print explain(plan)
print execute(plan)
