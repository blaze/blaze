"""
Script for performing queries on large time series by using UCR ED and DTW algs.
"""


from time import time
import os.path
import numpy as np
import blaze
from blaze.ts.ucr_dtw import ucr


# Convert txt file into Blaze native format
def convert(filetxt, storage):
    if not os.path.exists(storage):
        blaze.Array(np.loadtxt(filetxt),
                    params=blaze.params(storage=storage))

# Make sure that data is converted into a persistent Blaze array
convert("Data.txt", "Data")
convert("Query.txt", "Query")
convert("Query2.txt", "Query2")

t0 = time()
# Open Blaze arrays on-disk (will not be loaded in memory)
data = blaze.open("Data")
query = blaze.open("Query")
query2 = blaze.open("Query2")
print "Total Blaze arrays open time :", round(time()-t0, 4)

t0 = time()
# Do the search using the native Blaze format
#loc, dist = ucr.ed(data, query, 128)
loc, dist = ucr.dtw(data, query, 0.1, 128, verbose=False)
#loc, dist = ucr.dtw(data, query2, 0.1, 128)
t1 = time()

print "Location : ", loc
print "Distance : ", dist
print "Data Scanned : ", data.size
print "Total Execution Time :", round(time()-t0, 4)
