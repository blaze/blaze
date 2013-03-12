import time
import numpy as np
from blaze.blir import compile, Context, execute

#------------------------------------------------------------------------
# Matrix Assignement
#------------------------------------------------------------------------

source = """
def main(x: array[int], n : int) -> void {
    var int i;
    var int j;
    for i in range(n) {
        for j in range(n) {
            x[i,j] = i+j;
        }
    }
}
"""

N = 15
ast, env = compile(source)

arr = np.eye(N, dtype='int32')
args = (arr, N)

ctx = Context(env)
execute(ctx, args, timing=True)
ctx.destroy()

print arr

#------------------------------------------------------------------------
# Vector Dot Product
#------------------------------------------------------------------------

N = 50000
A = np.arange(N, dtype='double')
B = np.arange(N, dtype='double')

source = open('samples/blir/dot.bl')
ast, env = compile(source.read())

args = (A,B,N)

ctx = Context(env)
res = execute(ctx, args, fname='sdot', timing=True)
ctx.destroy()
print res

start = time.time()
print np.dot(A,B)
print 'Time %.6f' % (time.time() - start)
