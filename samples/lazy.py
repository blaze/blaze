from blaze import ones
from blaze.rts import execute
from blaze.compile import compile, explain

A = ones('100, 100, int32')
B = ones('100, 100, int32')

expr = (A + B) * B

# Normally we just use expr.eval() but to demonstrate the compile
# pipeline...

plan = compile(expr)

print explain(plan)
print execute(plan)
