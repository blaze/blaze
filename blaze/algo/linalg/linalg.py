########################################################################
#
#       Created: December 11, 2012
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

"""Different linear algebra functions.

The operations are being done with out-of-core algorithms.

"""

import sys, math

import numpy as np
import blaze


# The default size for OOC operation
OOC_BUFFER_SIZE = 2**25


def dot(a, b, out=None, outname='out'):
    """
    Matrix multiplication of two 2-D arrays.

    Parameters
    ----------
    a : array
        First argument.
    b : array
        Second argument.
    out : array, optional
        Output argument. This must have the exact kind that would be
        returned if it was not used. In particular, it must have the
        right type, must be C-contiguous, and its dtype must be the
        dtype that would be returned for `dot(a,b)`. This is a
        performance feature. Therefore, if these conditions are not
        met, an exception is raised, instead of attempting to be
        flexible.
    outname : str, optional
       If provided this will be the name for the output matrix storage.
       This parameter is only used when `out` is not provided.

    Returns
    -------
    output : array
        Returns the dot product of `a` and `b`.  If `a` and `b` are
        both scalars or both 1-D arrays then a scalar is returned;
        otherwise an array is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as the
        second-to-last dimension of `b`.
    """

    a_shape = tuple(i.val for i in a.datashape.parameters[:-1])
    b_shape = tuple(i.val for i in b.datashape.parameters[:-1])

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise (ValueError, "only 2-D matrices supported")

    if a_shape[1] != b_shape[0]:
        raise (ValueError,
               "last dimension of `a` does not match first dimension of `b`")

    l, m, n = a_shape[0], a_shape[1], b_shape[1]

    if out:
        out_shape = tuple(i.val for i in ouy.datashape.parameters[:-1])
        if out_shape != (l, n):
            raise (ValueError, "`out` array does not have the correct shape")
    else:
        parms = blaze.params(clevel=5, storage=outname)
        a_dtype = a.datashape.parameters[-1].to_dtype()
        dshape = blaze.dshape('%d, %d, %s' % (l, n, a_dtype))
        out = blaze.zeros(dshape, parms)


    # Compute a good block size
    out_dtype = out.datashape.parameters[-1].to_dtype()
    bl = math.sqrt(OOC_BUFFER_SIZE / out_dtype.itemsize)
    bl = 2**int(math.log(bl, 2))
    for i in range(0, l, bl):
        for j in range(0, n, bl):
            for k in range(0, m, bl):
                a0 = a[i:min(i+bl, l), k:min(k+bl, m)]
                b0 = b[k:min(k+bl, m), j:min(j+bl, n)]
                out[i:i+bl, j:j+bl] += np.dot(a0, b0)

    return out



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
