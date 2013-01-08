import numpy as np

def select(table, predicate, label):
    col = table.data.ca[label]
    nchunks = col.nchunks
    query = np.vectorize(predicate, otypes='b')

    for nchunk in range(nchunks):
        print query(col.chunks[nchunk][:])

def select2(table, predicate, labels):
    cols = [table.data.ca[label] for label in labels]
    nchunks = table.nchunks

    query = np.vectorize(predicate, otypes='b')

    for nchunk in range(nchunks):
        col1 = cols[0].chunks[nchunk][:]
        col2 = cols[1].chunks[nchunk][:]
        print query(col1, col2)
