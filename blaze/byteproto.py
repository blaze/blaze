#------------------------------------------------------------------------
# Hinting & Flags
#------------------------------------------------------------------------

CONTIGUOUS = 1
CHUNKED    = 2
STREAM     = 4
CHUNKED    = 8

READ  = 1
WRITE = 2
CONST = 4
READWRITE = READ | WRITE

LOCAL  = 1
REMOTE = 2
GPU = 2

ACCESS_ALLOC   = 1
ACCESS_READ    = 2
ACCESS_WRITE   = 4
ACCESS_COPY    = 8
ACCESS_APPEND  = 16
