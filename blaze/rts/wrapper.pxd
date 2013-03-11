cdef extern from "runtime.c":
    void *init_runtime(int nthreads)
    void join_runtime(void* rts)
