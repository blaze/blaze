#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------------------

typedef struct {
    char *data;
    int  nd;
    int  *strides;
} ndarray;

// ------------------------------------------------------------------------
// Math Operations
// ------------------------------------------------------------------------

int8_t abs_i8(int8_t i) {
  return i >= 0 ? i : -i;
}

int16_t abs_i16(int16_t i) {
  return i >= 0 ? i : -i;
}

int32_t abs_i32(int32_t i) {
  return i >= 0 ? i : -i;
}

int64_t abs_i64(int64_t i) {
  return i >= 0 ? i : -i;
}

// ------------------------------------------------------------------------
// String Operations
// ------------------------------------------------------------------------

int length(char* string)
{
    return strlen(string);
}

char indexof(char* string, int i)
{
    return string[i];
}

char* append(char* x, char* y)
{
    // leaks memory...
    char* buf = malloc((strlen(x)+strlen(y))*sizeof(char));
    strcpy(buf, x);
    strcat(buf, y);
    return buf;
}

int isNull(void* ptr)
{
    return ptr==NULL;
}

int strHead(char* str)
{
    if (str[0]=='\0') {
        return NULL;
    }
    return (int)(str[0]);
}

char* strTail(char* str)
{
    if (str[0]=='\0') {
        return NULL;
    }
    return str+1;
}

int strFind(char* str, char c)
{
    int i = 0;
    while(*str!='\0') {
        if (*str==c) {
            return i;
        }
        ++i;
        ++str;
    }
    return -1;
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

void show_array(ndarray *a) {
    printf("array(%p)", a->data);
}

#ifdef __cplusplus
}
#endif
