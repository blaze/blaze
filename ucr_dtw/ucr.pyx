#=================================================================
# Interface for the UCR suite (http://www.cs.ucr.edu/~eamonn/UCRsuite.html)
#
# Francesc Alted (francesc@continuum.io)
# Date: 2012-12-02
#
# Copyright © 2012 by Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen,
# Gustavo Batista and Eamonn Keogh.
#=================================================================

import sys
import os.path
from time import time
import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt, floor

# Some shortcuts for useful types
ctypedef np.npy_int npy_int
ctypedef np.npy_intp npy_intp
ctypedef np.npy_float64 npy_float64

#-----------------------------------------------------------------

# DTW routines
cdef extern from "dtw.h":

    void lower_upper_lemire(double *t, int len, int r, double *l, double *u)

    double lb_kim_hierarchy(double *t, double *q, int j, int len, double mean,
                            double std, double bsf)

    double lb_keogh_cumulative(int *order, double *t, double *uo, double *lo,
                               double *cb, int j, int len, double mean,
                               double std, double best_so_far)

    double lb_keogh_data_cumulative(int *order, double *tz, double *qo,
                                    double *cb, double *l, double *u,
                                    int len, double mean, double std,
                                    double best_so_far)

    double dtw_distance(double *A, double *B, double *cb, int m, int r,
                        double bsf)

#-----------------------------------------------------------------


# Initialize numpy library
np.import_array()


# Main function for calculating ED distance between the query, Q, and current
# data, T.  Note that Q is already sorted by absolute z-normalization value,
# |z_norm(Q[i])|
@cython.cdivision(True)
cdef double distance(double *Q, double *T, npy_intp j, int m, double mean,
                     double std, npy_intp *order, double bsf):
    cdef int i
    cdef double x, sum_ = 0
    cdef double istd = 1. / std
    for i in range(m):
        x = (T[order[i]+j] - mean) * istd
        sum_ += (x - Q[i]) * (x - Q[i])
        if sum_ >= bsf:
            break
    return sum_

@cython.boundscheck(False)
@cython.cdivision(True)
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
    cdef np.ndarray[npy_float64, ndim=1] Q    # query array
    cdef np.ndarray[npy_float64, ndim=1] T    # arrays of current data
    cdef np.ndarray[npy_intp, ndim=1] order   # ordering of query by |z(q_i)|
    cdef double bsf             # best-so-far
    cdef npy_intp loc = 0    # answer: location of the best-so-far match
    cdef npy_intp i, j
    cdef double d
    cdef int m, prevm
    cdef npy_intp fsize, nelements
    cdef double mean, std
    cdef double ex, ex2
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
    Q = (Q - mean) / std

    # Sort the query data
    order = np.argsort(np.abs(Q))
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
    # Prepare the ex and ex2 values for the inner loop (j) below
    LT = T[:m-1]
    ex = LT.sum()
    ex2 = (LT * LT).sum()

    bsf = np.inf
    # Read data file, m elements at a time
    for i in range(m, nelements, m):
        prevm = m
        if i + m > nelements:
            m = nelements - i
            # Recalculate the ex and ex2 values
            LT = T[:m-1]
            ex = LT.sum()
            ex2 = (LT * LT).sum()
        T[prevm:prevm+m] = np.fromfile(fp, 'f8', m)
        for j in range(m):
            # Update the ex and ex2 values
            ex += T[j+m-1]
            ex2 += T[j+m-1] * T[j+m-1]
            mean = ex / m
            std = ex2 / m
            std = sqrt(std - mean * mean)

            # Calculate ED distance
            dist = distance(<double*>Q.data, <double*>T.data, j,
                            m, mean, std, <npy_intp*>order.data, bsf)

            if dist < bsf:
                bsf = dist
                loc = (i - m) + j

            # Update the ex and ex2 values
            ex -= T[j]
            ex2 -= T[j] * T[j]

        # Copy the upper part of T to the lower part
        if prevm == m:
            T[:m] = T[m:]

    fp.close()
    dist = sqrt(bsf)
    t2 = time()

    print "Location : ", loc
    print "Distance : ", dist
    print "Data Scanned : ", i + m
    print "Time spent: %.3fs" % (time()-t0,)

    return (loc, dist)

# Main function for DTW
def dtw(datafile, queryfile, R, count=None):
    """Get the best DTW distance of `queryfile` in `datafile`.

    Parameters
    ----------
    `datafile` is the name of the file where the data to be queried is

    `queryfile` is the name of the file where the query data is

    `R` is the warping window (double)

    `count` is the number of elements to query (int).  If None, then the total
    length of the query array is used.

    Returns
    -------
    A tuple ``(loc, dist)`` where `loc` is the location of the best match,
    and `dist` is the distance.
    
    """
    cdef:
        double bsf          # best-so-far
        np.ndarray[npy_float64, ndim=1] q,t    # data array and query array
        np.ndarray[npy_int, ndim=1] order      # new order of the query
        np.ndarray[npy_float64, ndim=1] u, l, qo, uo, lo, tz, cb, cb1, cb2
        np.ndarray[npy_float64, ndim=1] u_d, l_d
        double d
        npy_intp i, j
        npy_intp loc = 0
        int m=-1, r=-1
        int kim = 0, keogh = 0, keogh2 = 0
        int it = 0, ep = 0, k = 0
        # For every EPOCH points, all cummulative values, such as ex (sum),
        # ex2 (sum square), will be restarted for reducing the floating point
        # error.
        int EPOCH = 100000
        double ex , ex2 , mean, std
        double t1, t2
        double dist=0, lb_kim=0, lb_k=0, lb_k2=0
        np.ndarray[npy_float64, ndim=1] buffer_, u_buff, l_buff
        npy_intp fsize, nelements, eleread, etoread
        # The starting index of the data in current chunk of size EPOCH
        npy_intp I
        object done = False
        object order_

    t1 = time()

    fsize = os.path.getsize(queryfile)
    nelements = fsize // np.dtype('f8').itemsize
    if count is None:
        m = nelements
    elif count > nelements:
        raise ValueError("count is larger than the values in queryfile")
    else:
        m = count

    if R <= 1:
        r = <int>floor(R*m)
    else:
        r = <int>floor(R)

    # Read the query data from input file and calculate its statistic such as
    # mean, std
    q = np.fromfile(queryfile, 'f8', m)
    mean = q.mean()
    std = q.std()

    # Do z_normalixation on query data
    q = (q - mean) / std

    # Create ancillary arrays here
    qo = np.empty(m, dtype="f8")
    uo = np.empty(m, dtype="f8")
    lo = np.empty(m, dtype="f8")
    u = np.empty(m, dtype="f8")
    l = np.empty(m, dtype="f8")
    cb = np.zeros(m, dtype="f8")
    cb1 = np.zeros(m, dtype="f8")
    cb2 = np.zeros(m, dtype="f8")
    u_d = np.empty(m, dtype="f8")
    l_d = np.empty(m, dtype="f8")
    t = np.empty(m*2, dtype="f8")
    tz = np.empty(m, dtype="f8")
    buffer_ = np.empty(EPOCH, dtype="f8")
    u_buff = np.empty(EPOCH, dtype="f8")
    l_buff = np.empty(EPOCH, dtype="f8")

    # Create envelop of the query: lower envelop, l, and upper envelop, u
    lower_upper_lemire(<double*>q.data, m, r, <double*>l.data, <double*>u.data)

    # Sort the query one time by abs(z-norm(q[i]))
    order_ = np.argsort(np.abs(q))
    # Reverse the order (from high to low) and convert to native ints
    order = order_[::-1].astype(np.intc)
    # Also create another arrays for keeping sorted envelop
    qo = q[order]
    uo = u[order]
    lo = l[order]

    bsf = np.inf
    i = 0          # current index of the data in current chunk of size EPOCH
    j = 0          # the starting index of the data in the circular array, t
    ex = ex2 = 0

    fsize = os.path.getsize(datafile)
    nelements = fsize // t.dtype.itemsize
    fp = open(datafile, 'r')
    eleread = 0

    while not done:
        # Protection agains queries larger than the input data
        if eleread + (m - 1) > nelements:
            m = nelements - eleread - 1

        # Read first m-1 points
        if it == 0:
            buffer_[:m-1] = np.fromfile(fp, 'f8', m-1)
            eleread += m-1  # XXX
        else:
            buffer_[:m-1] = buffer_[EPOCH-m+1:]

        # Read buffer of size EPOCH or when all data has been read.
        etoread = EPOCH - (m-1)
        if eleread + etoread > nelements:
            etoread = nelements - eleread
        buffer_[m-1:etoread+(m-1)] = np.fromfile(fp, 'f8', etoread)
        eleread += etoread
        ep = etoread + (m-1)

        # Data are read in chunk of size EPOCH.
        # When there is nothing to read, the loop is end.
        if ep <= (m-1):
            done = True
        else:
            lower_upper_lemire(<double*>buffer_.data, ep, r,
                               <double*>l_buff.data, <double*>u_buff.data)

            # Just for printing a dot for approximate a million point.  Not
            # much accurate.
            if (it % (1000000 / (EPOCH-m+1)) == 0):
                sys.stderr.write(".")

            # Do main task here.
            ex = ex2 = 0
            for i in range(ep):
                # A bunch of data has been read and pick one of them at a time
                # to use
                d = buffer_[i]

                # Calculate sum and sum square
                ex += d
                ex2 += d*d

                # t is a circular array for keeping current data
                t[i%m] = d

                # Double the size for avoiding using modulo "%" operator
                t[(i%m)+m] = d

                # Start the task when there are more than m-1 points in the
                # current chunk
                if i >= m-1:
                    mean = ex/m
                    std = ex2/m
                    std = sqrt(std - mean * mean)

                    # Compute the start location of the data in the current
                    # circular array, t
                    j = (i+1) % m
                    # The start location of the data in the current chunk
                    I = i-(m-1)

                    # Use a constant lower bound to prune the obvious
                    # subsequence
                    lb_kim = lb_kim_hierarchy(<double*>t.data, <double*>q.data,
                                              j, m, mean, std, bsf)

                    if lb_kim < bsf:
                        # Use a linear time lower bound to prune
                        # z_normalization of t will be computed on the fly.
                        # uo, lo are envelop of the query.
                        lb_k = lb_keogh_cumulative(
                            <int*>order.data, <double*>t.data,
                            <double*>uo.data, <double*>lo.data,
                            <double*>cb1.data, j, m, mean, std, bsf)
                        if lb_k < bsf:
                            # Take another linear time to compute
                            # z_normalization of t.  Note that for better
                            # optimization, this can merge to the previous
                            # function.
                            for k in range(m):
                                tz[k] = (t[(k+j)] - mean) / std

                            # Use another lb_keogh to prune. qo is the sorted
                            # query. tz is unsorted z_normalized data.
                            # l_buff, u_buff are big envelop for all data in
                            # this chunk.
                            lb_k2 = lb_keogh_data_cumulative(
                                <int*>order.data, <double*>tz.data,
                                <double*>qo.data, <double*>cb2.data,
                                <double*>l_buff.data + I,
                                <double*>u_buff.data + I,
                                m, mean, std, bsf)
                            if lb_k2 < bsf:
                                # Choose better lower bound between lb_keogh
                                # and lb_keogh2 to be used in early abandoning
                                # DTW Note that cb and cb2 will be cumulative
                                # summed here.
                                if lb_k > lb_k2:
                                    cb[m-1] = cb1[m-1]
                                    for k in range(m-2, -1, -1):
                                        cb[k] = cb[k+1]+cb1[k]
                                else:
                                    cb[m-1] = cb2[m-1]
                                    for k in range(m-2, -1, -1):
                                        cb[k] = cb[k+1]+cb2[k]

                                # Compute DTW and early abandoning if possible
                                dist = dtw_distance(
                                    <double*>tz.data, <double*>q.data,
                                    <double*>cb.data, m, r, bsf)

                                if dist < bsf:
                                    # Update bsf. loc is the real starting
                                    # location of the nearest neighbor in the
                                    # file
                                    bsf = dist
                                    loc = it * (EPOCH-m+1) + i-m+1
                            else:
                                keogh2 += 1
                        else:
                            keogh += 1
                    else:
                        kim += 1

                    # Reduce absolute points from sum and sum square
                    ex -= t[j]
                    ex2 -= t[j]*t[j]

            # If the size of last chunk is less then EPOCH, then no more data
            # and terminate.
            if ep < EPOCH:
                done = True
            else:
                it += 1

    i = it * (EPOCH-m+1) + ep
    fp.close()
    dist = sqrt(bsf)

    t2 = time()
    print "\n"

    # Note that loc and i are npy_intp
    print "Location : ", loc
    print "Distance : ", dist
    print "Data Scanned : ", i
    print "Total Execution Time : ", (t2-t1), "sec"

    print "\n"
    print "Pruned by LB_Kim    : %6.2f%%" % ((<double>kim / i)*100,)
    print "Pruned by LB_Keogh  : %6.2f%%" % ((<double>keogh / i)*100,)
    print "Pruned by LB_Keogh2 : %6.2f%%" % ((<double>keogh2 / i)*100,)
    print "DTW Calculation     : %6.2f%%" % (
        100-((<double>kim+keogh+keogh2)/i*100),)

    return (loc, dist)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
## End:
