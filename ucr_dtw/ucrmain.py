import os.path
import blaze
from ucr_dtw import ucr


# Convert txt file into Blaze native format
def convert(filetxt, storage):
    if not os.path.exists(storage):
        blaze.Array(np.loadtxt(filetxt),
                    params=blaze.params(storage=storage))

convert("Data.txt", "Data")
convert("Query.txt", "Query")
convert("Query2.txt", "Query2")

#ucr.ed("Data", "Query", 128)
#ucr.dtw("Data", "Query", 0.1, 128)
ucr.dtw("Data", "Query2", 0.1, 128)
