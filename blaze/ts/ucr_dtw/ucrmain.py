
import os.path
import numpy as np
import blaze
from blaze.ts.ucr_dtw import ucr


# Convert txt file into Blaze native format
def convert(filetxt, storage):
    if not os.path.exists(storage):
        blaze.Array(np.loadtxt(filetxt),
                    params=blaze.params(storage=storage))

convert("Data.txt", "Data")
convert("Query.txt", "Query")
convert("Query2.txt", "Query2")


# Do the search using the native Blaze format
#ucr.ed("Data", "Query", 128)
ucr.dtw("Data", "Query", 0.1, 128)
#ucr.dtw("Data", "Query2", 0.1, 128)
