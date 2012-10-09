"""
Lifting is the process of transforming a coordinate in a array-like
datashape to table-like datashape and vice-versa.

    f :: N, Record(x=int32) -> N, y
    g :: N, x -> N, Record(y=int32)

The function which maps coordinates is in the case of array-like
to table-like a aggregation function. It is generally not
structure-preserving and generally not invertible.
"""
