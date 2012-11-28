
import os.path
from time import time
import numpy as np
cimport numpy as np

# Initialize numpy library
np.import_array()

# Main function for calculating ED distance between the query, Q, and current
# data, T.  Note that Q is already sorted by absolute z-normalization value,
# |z_norm(Q[i])|
cdef double distance(double *Q, double *T, int m , double mean,
                     double std, int *order, double bsf):
    cdef int i
    cdef double x, sum_ = 0
    for i in range(m):
        x = (T[order[i]]-mean)/std
        sum_ += (x-Q[i])*(x-Q[i])
        if sum_ < bsf:
            return sum_
    return sum_


def ed(datafile, queryfile, count=None):
    """Get the best euclidean distance of `queryfile` in `datafile`.

    Parameters
    ----------
    `datafile` is the name of the file where the data to be queried is

    `queryfile` is the name of the file where the query data is

    `count` is the number of elements to query.  If None, then the total
    length of the query array is used.

    Returns
    -------
    A tuple ``(loc, dist)`` where `loc` is the location of the best match,
    and `dist` is the distance.
    
    """
    cdef np.ndarray Q           # query array
    cdef np.ndarray T, IT       # arrays of current data
    cdef np.ndarray order       # ordering of query by |z(q_i)|
    cdef double bsf             # best-so-far
    cdef np.npy_intp loc = 0    # answer: location of the best-so-far match
    cdef np.npy_intp i, j
    cdef double d
    cdef int m
    cdef np.npy_intp fsize, nelements
    cdef double mean, std
    cdef double dist = 0

    t0 = time()

    fsize = os.path.getsize(queryfile)
    nelements = fsize // np.dtype('f8').itemsize
    if count is None:
        m = nelements
    elif count > nelements:
        raise ValueError("count is larger than the values in queryfile")
    else:
        m = count

    # Read the query data from input file and calculate its statistic such as
    # mean, std
    Q = np.fromfile(queryfile, 'f8', m)
    mean = Q.mean()
    std = Q.std()

    # Do z_normalixation on query data
    for i in range(m):
         Q[i] = (Q[i] - mean)/std

    # Sort the query data
    order = np.argsort(Q)
    order = order[::-1].copy()   # reverse the order (from high to low)
    Q = Q[order]

    # Array for keeping the current data (it is twice the size for removing
    # modulo (circulation) in distance calculation)
    T = np.empty(2*m, dtype="f8")

    fsize = os.path.getsize(datafile)
    nelements = fsize // T.dtype.itemsize
    fp = open(datafile, 'r')

    # Bootstrap the process by reading the first m elements
    T[:m] = np.fromfile(fp, 'f8', m)

    bsf = np.inf
    # Read data file, m elements at a time
    for i in range(m, nelements, m):
        #print "i, m:", i, m
        if i + m > nelements:
            m = nelements - i
        T[m:] = np.fromfile(fp, 'f8', m)
        for j in range(m):
            IT = T[j:j+m]
            t1 = time()
            mean = IT.mean()
            std = IT.std()
            #print "time mean/std:", time()-t1
            # Calculate ED distance
            t1 = time()
            dist = distance(<double*>Q.data, <double*>IT.data,
                            m, mean, std, <int*>order.data, bsf)
            #print "time dist:", time()-t1
            if dist < bsf:
                bsf = dist
                loc = (i - m) + j +1
        # Copy the upper part of T to the lower part
        T[:m] = T[m:]
        # if i > 2*m:
        #     break

    fp.close()
    dist = np.sqrt(bsf)
    t2 = time()

    print "Location : ", loc
    print "Distance : ", dist
    print "Data Scanned : ", i + m
    print "Time spent: %.3fs" % (time()-t0,)

    return (loc, dist)

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
## End:
