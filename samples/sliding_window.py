"""
A simple demonstration of a sliding window implementation for Blaze.

The computation is performed with completely on-disk datasets.  After
the run a couple of datasets are created in the next directories:

Xp.blz: the original array over which the slicing window is computed
result.blz: the computed slicing window

You can have a peek at the underlying BLZ structure if you are curious (I hope you are ;)

Also, a comparison with NumPy results is done.
"""

import time
import numpy as np
import blaze as blz
import os, shutil

NROWS = 10000
WINDOW_SIZE = 100
timing = False


def sliding_window_numpy(X, window_size):
    filtered = np.empty_like(X)
    starts = window_size * [0] + range(1, NROWS - window_size + 1)
    for i in range(NROWS):
        start = starts[i]
        filtered[i] = (X[start:i + 1]).mean()
    return filtered


def sliding_window_blz(dirname, window_size):
    X = blz.open(dirname)
    if os.path.exists('result.blz'): shutil.rmtree('result.blz')
    filtered = blz.array([], dshape=X.datashape,
                         params=blz.params(storage='result.blz'))
    starts = window_size * [0] + range(1, NROWS - window_size + 1)
    for i in range(NROWS):
        start = starts[i]
        partial = (X[start:i + 1]).mean()
        filtered.append([partial])
    return filtered


if __name__ == '__main__':
    X = np.random.normal(0, 1, NROWS)
    #X = np.linspace(0, 1, NROWS)

    start = time.time()
    result_numpy = sliding_window_numpy(X, WINDOW_SIZE)
    if timing: print 'numpy', time.time() - start

    if os.path.exists('Xp.blz'): shutil.rmtree('Xp.blz')
    Xp = blz.array(X, params=blz.params(storage='Xp.blz'))

    start = time.time()
    result_blaze = sliding_window_blz('Xp.blz', WINDOW_SIZE)
    if timing: print 'blaze', time.time() - start

    print "numpy result", result_numpy
    print "blaze result", result_blaze
    print "allclose?", np.allclose(result_numpy, result_blaze[:])
