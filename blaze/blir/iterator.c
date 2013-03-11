#include "Python.h"

// ------------------------------------------------------------------------

enum {
    BUFFER,
    STREAM,
    BUFFERLIST,
    STREAMLIST,
} iter_t;

typedef struct {
} bufiter;

typedef struct {
} streamiter;

typedef struct {
    int chunk;
    int chunklen;
    int nchunks;
} buflistiter;

typedef struct {
} streamlistiter;

typedef struct {
    int tag;
    union {
        bufiter bit;
        streamiter sit;
        buflistiter blit;
        streamlistiter slit;
    } iter;
} bliter;

// ------------------------------------------------------------------------

typedef struct {
    char *data;
    int  nd;
    int  *strides;
} ndarray;

// ------------------------------------------------------------------------

typedef struct {
    int (*next) (void *desc, bliter *);
} datadescriptor;

// ------------------------------------------------------------------------

int carray_next(PyObject *ca, bliter *it, ndarray *buf)
{
    int finished = 0;
    buflistiter bl = it->iter.blit;

    PyObject *chunk;
    PyObject *chunks;

    int data_ptr;

    chunks = PyObject_GetAttrString(ca, "chunks");

    if (bl.chunk < bl.nchunks) {
        // Load the chunk from disk ...
        chunk = PyList_GET_ITEM(chunks, bl.chunk);
        data_ptr = PyLong_AsLong(PyObject_GetAttrString(chunk, "pointer"));
        buf->data = (char*)data_ptr;
    } else if (bl.chunk == bl.nchunks) {
        data_ptr = PyLong_AsLong(PyObject_GetAttrString(ca, "leftovers"));
        buf->data = (char*)data_ptr;
    } else {
        finished = 1;
    }

    return finished;
}

void carray_seek(PyObject *ca, bliter *it)
{
}

void carray_commt(PyObject *ca, bliter *it)
{
}

void carray_done(PyObject *ca, bliter *it)
{
}

// ------------------------------------------------------------------------
