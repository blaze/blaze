#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_THREADS 128

typedef void (*kernel_t) (int, int, void* args);

typedef struct {
    int nthreads;
    pthread_t threads[MAX_THREADS];
} runtime_t;

typedef struct {
    kernel_t     kernel;
    void        *args;
    volatile int done;
} context_t;

void worker(context_t *context)
{
}

runtime_t *init_runtime(int nthreads)
{
    int i;
    runtime_t *rts = (runtime_t*)malloc(sizeof(runtime_t));
    rts->nthreads = nthreads;

    for (i = 0; i < rts->nthreads; i++ ) {
        printf("Spawning threads %i\n", i);
        pthread_create(&rts->threads[i], NULL, worker, NULL);
    }

    return rts;
}

void destroy_runtime(runtime_t *rts)
{
    int i;

    for (i = 0; i < rts->nthreads; i++ ) {
        pthread_cancel(rts->threads[i]);
    }

    free(rts);
}

void join_runtime(runtime_t *rts)
{
    int i;

    for (i = 0; i < rts->nthreads; i++ ) {
        printf("Joining threads %i\n", i);
        pthread_join(rts->threads[i], NULL);
    }
}


