"""
Cython bindings for libAterm, a small no-dependency C library for
manipulating and parsing ATerm expressions.

Docs
----

* http://strategoxt.org/Tools/ATermLibrary
* http://www.meta-environment.org/doc/books/technology/aterm-guide/aterm-guide.html

"""

cdef extern from "stdarg.h":
    ctypedef struct va_list:
        pass
    ctypedef struct fake_type:
        pass
    void va_start(va_list, void* arg)
    void* va_arg(va_list, fake_type)
    void va_end(va_list)
    fake_type int_type "int"

cdef extern from "stdio.h":
    ctypedef FILE
    enum: stdout

cdef extern from "aterm1.h":
    enum: AT_FREE
    enum: AT_APPL
    enum: AT_INT
    enum: AT_REAL
    enum: AT_LIST
    enum: AT_PLACEHOLDER
    enum: AT_BLOB
    enum: AT_SYMBOL
    enum: MAX_ARITY

    ctypedef long MachineWord
    ctypedef unsigned long HashNumber
    ctypedef unsigned long header_type

    ctypedef struct __ATerm:
        header_type header
        ATerm *next

    ctypedef union ATerm:
        header_type header
        __ATerm aterm
        ATerm* subaterm[MAX_ARITY+3]
        MachineWord  word[MAX_ARITY+3]

    ctypedef int *FILE

    void ATinit (int argc, char *argv[], ATerm *bottomOfStack)
    ATerm ATmake(char *pattern, ...)
    ATbool ATmatch(ATerm t, char *pattern, ...)

    int ATprintf(char *format, ...)
    int ATfprintf(int stream, char *format, ...)
    char *ATwriteToString(ATerm t)

    ATerm ATreadFromString(char *string)
    ATerm ATreadFromSharedString(char *s, int size)

    ATerm ATsetAnnotation(ATerm t, ATerm label, ATerm anno)
    ATerm ATgetAnnotation(ATerm t, ATerm label)

    void ATsetWarningHandler(void (*handler)(char *format, va_list args))
    void ATsetErrorHandler(void (*handler)(char *format, va_list args))
    void ATsetAbortHandler(void (*handler)(char *format, va_list args))

    ctypedef enum ATbool:
        ATfalse = 0
        ATtrue  = 1

    ATerm ATisEqual(ATerm t1, ATerm t2)

cdef ATerm ATEmpty

cdef class PyATerm:
    cdef ATerm a
    cdef char* _repr

    def __init__(self, pattern):
        cdef ATerm a = ATreadFromString(pattern)

        if a == ATEmpty:
            raise Exception('Invalid ATerm: %s' % pattern)
        else:
            self.a = a
            self._repr = ATwriteToString(self.a)

    def __setitem__(self, char* key, char* value):
        cdef ATerm label = ATreadFromString(key)
        cdef ATerm anno = ATreadFromString(value)
        self.a = ATsetAnnotation(self.a, label, anno)

    def __getitem__(self, char* key):
        cdef ATerm label = ATreadFromString(key)
        cdef ATerm value = ATgetAnnotation(self.a, label)
        if value == ATEmpty:
            raise KeyError(key)
        else:
            return ATwriteToString(value)

    #def __richcmp__(self, other, int op):
        #cdef ATbool res
        #res = ATmatch(self.a, pattern)

       #if op == 2:
           #if isinstance(other, PyATerm):
               #res = ATisEqual(self.a, other.a)

        #if res == ATtrue:
            #return True
        #if res == ATfalse:
            #return False

    def matches(self, char* pattern):
        """
        Matches against ATerm patterns.

        >>> ATerm('x').matches('<term>')
        # True

        Alas, no metadata annotation ... at least out of the box.
        Ergo the reason for my half baked query language. Think
        I can roll it in here though

        """
        cdef ATbool res
        res = ATmatch(self.a, pattern)

        if res == ATtrue:
            return True
        if res == ATfalse:
            return False

    def __repr__(self):
        return "aterm('%s')" % ATwriteToString(self.a)

cdef void error(char *format, va_list args) with gil:
    raise Exception(format)

# -- init --

cdef ATerm bottomOfStack
ATinit(1, [], &bottomOfStack)

# Register error handlers
ATsetErrorHandler(error)
ATsetWarningHandler(error)
ATsetAbortHandler(error)

# -- toplevel --

def aterm(str s):
    return PyATerm(s)

def matches(str s, PyATerm term):
    return term.matches(s)
