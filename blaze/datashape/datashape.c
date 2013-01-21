#include "Python.h"
#include "structmember.h"

#ifndef Py_TYPE
#  define Py_TYPE(ob)   (((PyObject *) (ob))->ob_type)
#endif

#ifdef DEBUG
#   define TRACE()    printf("%s:%s:%d\n",__FILE__,__FUNCTION__,__LINE__)
#else
#   define TRACE()    ((void) 0)
#endif

#ifdef DEBUG
#    define MALLOC PyMem_Malloc
#    define FREE PyMem_FREE
#else
#    define MALLOC malloc
#    define FREE free
#endif

/*---------------------------------------------------------*/

typedef enum
{
   NIL, PARAM, TYPEVAR, DYNAMIC, CTYPE, FIXED, PRODUCT, RANGE,
   EITHER, OPTION, UNION, FACTOR, RECORD
} kind_t;

#define kindof(t) (kind_t)((type_t*)(t)->kind)

/*---------------------------------------------------------*/
struct type_t;
typedef struct type_t TYPE;

typedef struct type_t
{
   kind_t kind;         /* kind of type */
   int nparams;         /* number of parameters */
   int *params;         /* type parameters */
   int index;           /* de Bruijn index of a TypeVar */
   TYPE *body;          /* body of a parameterized type */
   TYPE **args;         /* arguments of a construct */
} type_t;

/*---------------------------------------------------------*/

type_t *null_type(void)
{
    type_t * t = (type_t *) MALLOC(sizeof(type_t));
    t->kind = NIL;
    return t;
}

type_t *dynamic_type(void)
{
    type_t *t = (type_t *) MALLOC(sizeof(type_t));
    t->kind = DYNAMIC;
    return t;
}

type_t *typevar_type(int index)
{
    type_t *t = (type_t *) MALLOC(sizeof(type_t));
    t->kind = TYPEVAR;
    t->index = index;
    return t;
}

type_t *param_type(int n, int* indices, type_t * body)
{
    type_t *t = (type_t *) MALLOC(sizeof(type_t));
    t->kind = PARAM;
    t->nparams = n;
    t->params = indices;
    t->body = body;
    return t;
}

type_t *product_type(int n, type_t ** args)
{
    type_t *t = (type_t *) MALLOC(sizeof(type_t));
    t->kind = PRODUCT;
    t->nparams = n;
    t->args = args;
    return t;
}

void type_free(struct type_t *t)
{
    assert(t != NULL);
    free(t->params);
    free(t->body);
    free(t->args);
    free(t);
}

/*---------------------------------------------------------*/

typedef struct {
    PyObject_HEAD
    TYPE *ty;
} DshapeObject;

/*---------------------------------------------------------*/

static PyObject *
PyDshape_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    DshapeObject *self;
    self = (DshapeObject *)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->ty = NULL;
    }

    return (PyObject *)self;
}

static int
Dshape_init(DshapeObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static void
Dshape_dealloc(DshapeObject* self)
{
    #ifndef DEBUG
    type_free(self->ty);
    #endif
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
Dshape_kind(DshapeObject *self)
{
    if(self->ty == NULL) {
        return Py_None;
    }

    return PyInt_FromLong(self->ty->kind);
}

/*---------------------------------------------------------*/

static PyMemberDef Dshape_members[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef Dshape_methods[] = {
    {"kind", (PyCFunction)Dshape_kind, METH_NOARGS , NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyDshapeType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "cdatashape.dshape",       /*tp_name*/
    sizeof(DshapeObject),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Dshape_dealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "dshape",                  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Dshape_methods,            /* tp_methods */
    Dshape_members,            /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Dshape_init,     /* tp_init */
    0,                         /* tp_alloc */
    PyDshape_new,              /* tp_new */
};

/*---------------------------------------------------------*/

#define Dshape_SET(ds, t) ((DshapeObject *)ds)->ty = t;

int
Dshape_Check(PyObject *ds)
{
    return (Py_TYPE(ds) == &PyDshapeType);
}

int
Dshape_Set(PyObject *ds, type_t *t)
{
    if (Dshape_Check(ds)) {
        ((DshapeObject *)ds)->ty = t;
        return 1;
    } else {
        return 0;
    }
}

static PyObject *
Py_Nil(PyObject *self)
{
    PyObject *ds;
    type_t *t = null_type();

    PyObject *args = PyTuple_New(0);
    ds = PyObject_CallObject((PyObject *) &PyDshapeType, args);
    Dshape_Set(ds, t);

    return (PyObject*)ds;
}

static PyObject *
Py_Dynamic(PyObject *self)
{
    PyObject *ds;
    type_t *t = dynamic_type();

    PyObject *args = PyTuple_New(0);
    ds = PyObject_CallObject((PyObject *) &PyDshapeType, args);
    Dshape_Set(ds, t);

    return (PyObject*)ds;
}

static PyObject *
Py_TypeVar(PyObject *self, PyObject *args)
{
    PyObject *ds;
    int index;
    type_t *t;

    if (!PyArg_ParseTuple(args, "i", &index)) {
        PyErr_SetString(PyExc_TypeError, "Invalid de brujin index");
    }
    ds = PyObject_CallObject((PyObject *) &PyDshapeType, args);

    t = typevar_type(index);
    Dshape_Set(ds, t);

    return ds;
}

static PyObject *
Py_Product(PyObject *self, PyObject *args)
{
    int i;
    PyObject *ix;
    PyObject *lst;

    if (!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    assert(PyList_Check(lst));

    for (i = 0; i < PyList_Size(lst); i++) {
        ix = PyList_GetItem(lst, (Py_ssize_t)i);

        if (PyInt_Check(ix)) {
            PyErr_SetString(PyExc_TypeError, "Not 'int'");
            return NULL;
        }
        PyInt_AsSsize_t(ix);
    }
    return NULL;
}

/*---------------------------------------------------------*/

static PyMethodDef functions[] = {
    {"nil", (PyCFunction) Py_Nil, METH_NOARGS, NULL} ,
    {"dynamic", (PyCFunction) Py_Dynamic, METH_NOARGS, NULL} ,
    {"typevar", (PyCFunction) Py_TypeVar, METH_VARARGS, NULL} ,
    {"product", (PyCFunction) Py_Product, METH_VARARGS, NULL} ,
    {NULL, NULL}
};

/*---------------------------------------------------------*/

void initcdatashape(void)
{
    PyObject* m;

    m = Py_InitModule("cdatashape", functions);

    if (PyType_Ready(&PyDshapeType) < 0) {
        return;
    }

    if (m == NULL) {
        return;
    }

    Py_INCREF(&PyDshapeType);
    PyModule_AddObject(m, "dshape", (PyObject *)&PyDshapeType);
}
