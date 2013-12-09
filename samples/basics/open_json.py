'''Sample module showing how to read JSON files into blaze arrays'''

import blaze
from blaze.datadescriptor import dd_as_py
#from dynd import nd

json_buf = u"[1, 2, 3, 4, 5]"
json_schema = "var, int8"

# Create a temporary JSON file in URI and open the dataset
dname = 'json:///tmp/test.json'
store = blaze.Storage(dname)
print "store:", store
with file(store.path, "wb") as f:
    f.write(json_buf)
arr = blaze.open(store, json_schema)

#print('Blaze array:', arr)  # XXX This does not work yet
#print('Blaze array:', nd.array(arr))  # XXX idem
# Convert the data to a native Python object
print('Blaze array:', dd_as_py(arr._data))

# Remove the temporary JSON file
blaze.drop(store)
