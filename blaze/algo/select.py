import numpy as np

def select(table, predicate, label):
    col = table.data.ca[label]
    nchunks = col.nchunks
    query = np.vectorize(predicate)
    qs = list()

    for nchunk in range(nchunks):
        qi = list(query(col.chunks[nchunk][:]))
        qs.append(qi)

    return qs

def select2(table, predicate, labels):
    cols = [table.data.ca[label] for label in labels]
    nchunks = cols[0].nchunks

    query = np.vectorize(predicate)
    qs = list()

    for nchunk in range(nchunks):
        col1 = cols[0].chunks[nchunk][:]
        col2 = cols[1].chunks[nchunk][:]

        qi = list(query(col1, col2))
        qs.append(qi)

    return qs
