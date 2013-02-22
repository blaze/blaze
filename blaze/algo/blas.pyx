# compile with lcblas

cimport numpy as np
np.import_array()

# S single real
# D double real
# C single complex
# Z double complex

#------------------------------------------------------------------------
# _scal
#------------------------------------------------------------------------


cdef void sscal(float alpha, np.ndarray x):
    cblas_sscal(x.shape[0], alpha, <float*>x.data, 1)

cdef void dscal(double alpha, np.ndarray x):
    cblas_dscal(x.shape[0], alpha, <double*>x.data, 1)

#------------------------------------------------------------------------
# _dot
#------------------------------------------------------------------------

cdef float sdot(np.ndarray x, np.ndarray y):
    return cblas_sdot(x.shape[0], <float*>x.data, 1, <float*>y.data, 1)

cdef float ddot(np.ndarray x, np.ndarray y):
    return cblas_ddot(x.shape[0], <double*>x.data, 1, <double*>y.data, 1)


#------------------------------------------------------------------------
# _gemm
#------------------------------------------------------------------------

cdef np.ndarray sgemm(np.ndarray A, np.ndarray B):
    cdef:
        np.ndarray C = np.ndarray(A.shape[0], B.shape[1])
        float alpha = 1.0
        float beta = 0.0

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.shape[0],
                C.shape[1], B.shape[0], alpha, <float*>A.data, A.shape[1],
                <float*>B.data, B.shape[1], beta, <float*>C.data, C.shape[1])
    return C


cdef np.ndarray dgemm(np.ndarray A, np.ndarray B):
    cdef:
        np.ndarray C = np.empty((A.shape[0], B.shape[1]))
        double alpha = 1.0
        double beta = 0.0

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.shape[0],
                C.shape[1], B.shape[0], alpha, <double*>A.data, A.shape[1],
                <double*>B.data, B.shape[1], beta, <double*>C.data, C.shape[1])
    return C
