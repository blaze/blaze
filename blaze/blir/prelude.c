#include <stdio.h>
#include <string.h>
#include <malloc.h>

// Later...
//#include <blosc.h>
//#include <zmq.h>

//#include <mkl.h>
//#include <mkl_types.h>
//#include <mkl_lapack.h>
//#include <mkl_cblas.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "datashape.h"

// ------------------------------------------------------------------------
// Execution State
// ------------------------------------------------------------------------

static int initialized = 0;

void Blir_Initialize()
{
    if (initialized) {
        return;
    }
    initialized = 1;
}

void Blir_Finalize()
{
    if (!initialized) {
        return;
    }
    initialized = 0;
}

// ------------------------------------------------------------------------
// Datashape Operations
// ------------------------------------------------------------------------

// XXX Very limited for now...

void is_fixed(type_t *ds)
{
    kindof(ds) == FIXED;
}

void is_ctype(type_t *ds)
{
    kindof(ds) == CTYPE;
}

// ------------------------------------------------------------------------
// String Operations
// ------------------------------------------------------------------------

int isNull(void* ptr)
{
    return ptr==NULL;
}

int length(char* string)
{
    return strlen(string);
}

char charAt(char* string, int i)
{
    return string[i];
}

char* append(char* x, char* y)
{
    char* buf = malloc((strlen(x)+strlen(y))*sizeof(char));
    strcpy(buf, x);
    strcat(buf, y);
    return buf;
}

// ------------------------------------------------------------------------
// Printing
// ------------------------------------------------------------------------

void show_int(int i)
{
    printf("%i\n", i);
}

void show_float(double f)
{
    printf("%f\n", f);
}

void show_string(char* s)
{
    printf("%s\n", s);
}

void show_bool(int b)
{
    if (b) {
        printf("True\n");
    } else {
        printf("False\n");
    }
}

#ifdef __cplusplus
}
#endif
