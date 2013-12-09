'''Sample module showing how to read CSV files into blaze arrays'''

import blaze
from blaze.datadescriptor import dd_as_py


# A CSV toy example
csv_buf = u"""k1,v1,1,False
k2,v2,2,True
k3,v3,3,False
"""
csv_schema = "{ f0: string; f1: string; f2: int16; f3: bool }"

# Create a CSV file in URI and open the dataset
dname = 'csv:///tmp/test.csv'
store = blaze.Storage(dname)
print "store:", store
with file(store.path, "wb") as f:
    f.write(csv_buf)
arr = blaze.open(store, csv_schema)

#print('Blaze array:', arr)  # XXX This does not work yet
#print('Blaze array:', nd.array(arr))  # XXX idem
# Convert the data to a native Python object
print('Blaze array:', dd_as_py(arr._data))

# Remove the CSV file
blaze.drop(store)
