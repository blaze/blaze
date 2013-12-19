===================================
Access Data from Different Sources
===================================

Blaze can access data from a variety of sources, as long as the format has been implemented as a :ref:`Data Descriptor <data_descriptor>`.  Currently there are data descriptors for CSV and JSON files, although the list is growing.  This section shows how to access these files in a general way.

Open and working with a data file
---------------------------------

Let's suppose that we have a file named '/tmp/test.csv' that we want to operate with in Blaze.  Blaze normally access data through URLs, so first, let's use the the Storage class so as to parse the URL and determine if we can process the file::

  In []: store = blaze.Storage('csv:///tmp/test.csv')

The first part of the URL, the network protocol, specifies the format of the data source, and the rest tells where the data is.  In this case we are parsing a CSV file, so this is why the network protocol is 'csv'.  For JSON files it is just a matter of replacing this part by 'json' instead of 'csv'.

Now, for actually accessing the data in the file we need to know the schema of the files, let's use the `open` function on our `store` instance::

  In []: csv_schema = "{ f0: string; f1: string; f2: int16; f3: bool }"
  In []: arr = blaze.open(store, schema=csv_schema)

As we see, the `open` function needs you to inform about the schema of the underlying file; in this case, each line is formed by a couple of strings, an integer of 16 bits and a boolean.

If we want to have a look at the contents, then just print the Blaze array:: 

  In []: arr._data.dynd_arr()  # update this when struct types can be pri
  Out[]: nd.array([["k1", "v1", 1, false], ["k2", "v2", 2, true], ["k3", "v3", 3, false]], var_dim<{f0 : string; f1 : string; f2 : int16; f3 : bool}>)

