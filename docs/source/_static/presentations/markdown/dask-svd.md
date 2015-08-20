## Data Ingest / JSON Blob Computations

    >>> import dask.bag as db
    >>> b = db.from_s3('githubarchive-data', '2015-*.json.gz')
              .map(json.loads)
              .map(lambda d: d['type'] == 'PushEvent')
              .count()

[*Problem and blogpost by Blake Griffith*](continuum.io/blog/dask-distributed-cluster)


## Example: SVD


## What about more complex workflows?

    >>> import dask.array as da
    >>> x = da.ones((5000, 1000), chunks=(1000, 1000))
    >>> u, s, v = da.linalg.svd(x)

<a href="images/dask-svd.png">
  <img src="images/dask-svd.png" alt="Dask SVD graph" width="30%">
</a>

*Work by Mariano Tepper.  "Compressed Nonnegative Matrix Factorization is Fast
and Accurate" [arXiv](http://arxiv.org/abs/1505.04650)*


## SVD - Dict

    >>> s.dask
    {('x', 0, 0): (np.ones, (1000, 1000)),
     ('x', 1, 0): (np.ones, (1000, 1000)),
     ('x', 2, 0): (np.ones, (1000, 1000)),
     ('x', 3, 0): (np.ones, (1000, 1000)),
     ('x', 4, 0): (np.ones, (1000, 1000)),
     ('tsqr_2_QR_st1', 0, 0): (np.linalg.qr, ('x', 0, 0)),
     ('tsqr_2_QR_st1', 1, 0): (np.linalg.qr, ('x', 1, 0)),
     ('tsqr_2_QR_st1', 2, 0): (np.linalg.qr, ('x', 2, 0)),
     ('tsqr_2_QR_st1', 3, 0): (np.linalg.qr, ('x', 3, 0)),
     ('tsqr_2_QR_st1', 4, 0): (np.linalg.qr, ('x', 4, 0)),
     ('tsqr_2_R', 0, 0): (operator.getitem, ('tsqr_2_QR_st2', 0, 0), 1),
     ('tsqr_2_R_st1', 0, 0): (operator.getitem,('tsqr_2_QR_st1', 0, 0), 1),
     ('tsqr_2_R_st1', 1, 0): (operator.getitem, ('tsqr_2_QR_st1', 1, 0), 1),
     ('tsqr_2_R_st1', 2, 0): (operator.getitem, ('tsqr_2_QR_st1', 2, 0), 1),
     ('tsqr_2_R_st1', 3, 0): (operator.getitem, ('tsqr_2_QR_st1', 3, 0), 1),
     ('tsqr_2_R_st1', 4, 0): (operator.getitem, ('tsqr_2_QR_st1', 4, 0), 1),
     ('tsqr_2_R_st1_stacked', 0, 0): (np.vstack,
                                       [('tsqr_2_R_st1', 0, 0),
                                        ('tsqr_2_R_st1', 1, 0),
                                        ('tsqr_2_R_st1', 2, 0),
                                        ('tsqr_2_R_st1', 3, 0),
                                        ('tsqr_2_R_st1', 4, 0)])),
     ('tsqr_2_QR_st2', 0, 0): (np.linalg.qr, ('tsqr_2_R_st1_stacked', 0, 0)),
     ('tsqr_2_SVD_st2', 0, 0): (np.linalg.svd, ('tsqr_2_R', 0, 0)),
     ('tsqr_2_S', 0): (operator.getitem, ('tsqr_2_SVD_st2', 0, 0), 1)}


## SVD - Parallel Profile

<iframe src="svd.profile.html"
        marginwidth="0"
        marginheight="0" scrolling="no" width="800"
        height="300"></iframe>

*Bokeh profile tool by Jim Crist*


## Randomized Approximate Parallel Out-of-Core SVD

    >>> import dask.array as da
    >>> x = da.ones((5000, 1000), chunks=(1000, 1000))
    >>> u, s, v = da.linalg.svd_compressed(x, k=100, n_power_iter=2)

<a href="images/dask-svd-random.png">
<img src="images/dask-svd-random.png"
     alt="Dask graph for random SVD"
     width="10%" >
</a>

N. Halko, P. G. Martinsson, and J. A. Tropp.
*Finding structure with randomness: Probabilistic algorithms for
constructing approximate matrix decompositions.*

*Dask implementation by Mariano Tepper*
