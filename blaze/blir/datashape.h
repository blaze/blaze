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
