import warnings

cdef extern from "pthread.h" nogil:
    enum: PTHREAD_CANCELED

    ctypedef unsigned long int pthread_t
    ctypedef union pthread_attr_t:
        char __size[56]
        long int __align

    int pthread_create(pthread_t *thread, pthread_attr_t *attr,
         void *(*start_routine) (void *), void *arg)
    int pthread_join(pthread_t thread, void **retval)
    int pthread_cancel (pthread_t __th)

cdef class OSThread:

    cdef public pthread_t thread
    cdef public int retval
    cpdef running

    def __cinit__(self, pthread_t thread, *res):
        self.thread = thread
        self.running = True

    def __dealloc__(self):
        self.ensure()

    def join(self):
        if self.running:
            pthread_cancel(self.thread)
            self.running = False
        return self.retval

    def ensure(self):
        if not self.running:
            return

        warnings.warn('Forcefully destroying io thread!', RuntimeWarning)
        rt = pthread_cancel(self.thread)
        rc = self.join()
        if rc != 0:
            raise RuntimeError("IO Thread did not exit cleanly: rc = %i" % rc) 

    def __repr__(self):
        return "<OSThread(%s, %s)>" % (self.thread, self.running)
