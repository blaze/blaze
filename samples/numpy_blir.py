import numpy as np
from blaze.blir import compile, execute

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

execute(env, args, timing=True)
print arr
