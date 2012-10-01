"""
Runtime ensures that the data is available in-core when the algorithm needs it.
"""

#    Action |  Local Pointer | hint
#
# 1) Load A -> ptr 0x92e7b8    !contigious
# 2) Load B -> ptr 0x923910    !contigious
# 3) Load C -> ptr 0x92e2a0    !contigious
# 4) Execute ufunc ut.matrix_multiply ( 0x92e7b8, 0x923910 ) -> ptr 0x923101
# 5) Execute ufunc ut.matrix_multiply ( 0x923101, 0x92e2a0 ) -> ptr 0x923f00

class Schedule(object):
    pass

# Fetch all the data at once, store it all into memory, and
# perform all operations in serial.
class NaiveSchedule(Schedule):
    pass

# The sufficiently clever runtime would be able to parallelize,
# operations and execute remote computations efficiently when possible
# given its knowledge of the data types in question.

# Goal would not be to actually implement this in Python but to
#   a) have an execution model that is simple to reason about
#   b) translates easily into to C or LLVM
#   c) a way to explore parallelization

class Buffer(object):
    pass

class Stream(object):
    pass

class BufferList(object):
    pass

class StreamList(object):
    pass

# BUFFERPOP     [ TOS, ]
#                    ^
#                    |
# +---+---+---+---+-----+
# | D | C | B | A | TOS |
# +---+---+---+---+-----+
#                    |
#                    |
#                    v
# STREAMPOP     Stream : TOS

opcodes = [
    # These would C functions provided by the adaptor. copy bytes
    # from a byte-source or to a byte-sink.

    # Examples:
    # * memcpy(dest, src, nbytes)
    # * read(fd, dest, nbytes)
    # * zmq_recvmsg(sock, src, nbytes);
    'BYTESFROM',

    # Examples:
    # * write(fd, src, nbytes)
    # * zmq_send(sock, src, nbytes);
    'BYTESTO',

    # Allocate local memory data storage, will be able to understand
    # datashape objects of native types like (2*3*int32) literally.
    # Returns the pointer to a register for use.
    'MALLOC',
    'FREE',

    # Take a Numpy/Numba ufunc object provide it with pointers to
    # arguments, push the resulting pointer on the stack.
    'UFUNC',

    # Take a Numba CFunction object and provide it with pointer
    # to arguments, push the resulting pointer on the stack.
    'CFUNC',

    # Store contents of top of stack at the address
    'STORE',

    # Removes the top of stack as a buffer, pushes it into a
    # StreamList instance.
    'STREAMPOP',

    # Removes the top of stack as a buffer, pushes it into a
    # BufferList instance.

    'BUFFERPOP',
]
