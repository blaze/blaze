#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

#include <blosc.h>

#include "worker.h"

#define MAX_THREADS 5

typedef struct {
    int pid;
    int nthreads;
    pthread_t threads[MAX_THREADS];
} Runtime;


typedef struct {
    void *vartable;
    void *instructions;
    unsigned id;
} Context;


Runtime *init_runtime(int nthreads) {
    unsigned i;
    Runtime *rts = (Runtime*)malloc(sizeof(Runtime));
    Context contexts[nthreads];

    if(nthreads > MAX_THREADS) {
        fprintf(stderr, "ERROR; Too many threads");
    }

    for (i=0; i < rts.nthreads; ++i){
        contexts[i].id = i;
        pthread_create(&threads[i], NULL, worker, (void*)&contexts[i]);
    }
}

void destroy_runtime(Runtime *rts) {
    unsigned i;

    for (i = 0; i < rts.nthreads; ++i ){
        pthread_cancel(rts->threads[i]);
    }

    free(rts);
}

void join_runtime(Runtime *rts) {
    unsigned i;

    for (i = 0; i < rts.nthreads; ++i ){
        pthread_join(rts->threads[i], NULL);
    }
}

void worker(Context *context) {
}
