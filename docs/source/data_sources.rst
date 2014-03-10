===================================
Access Data from Different Sources
===================================

Blaze can access data from a variety of sources, as long as the format has been implemented as a :ref:`Data Descriptor <data_descriptor>`.  Currently there are data descriptors for DyND, BLZ, HDF5, CSV and JSON files, although the list is growing.  This section shows how to access these files in a general way.

Working with BLZ storage
------------------------

BLZ is a storage solution that allows to keep your Blaze datasets either in-memory or on-disk.

In-memory storage:

.. doctest::

  >>> blz = blaze.array([1,2,3], caps={'compress': True})
  >>> print(blz)
  [1 2 3]

It is possible to force the type when creating the array. This
allows a broader selection of types on construction:

.. doctest::

  >>> e = blaze.array([ 1, 2, 3], dshape='3, float32') 
  >>> e
  array([ 1.,  2.,  3.],
        dshape='3, float32')

Note that the dimensions in the datashape when creating from a
collection can be omitted. If that's the case, the dimensions will be
inferred. The following is thus equivalent:

.. doctest::

  >>> f = blaze.array([ 1, 2, 3], dshape='float32')
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3, float32')

BLZ also supports arrays to be made persistent. This can be achieved
by adding the storage keyword parameter to an array constructor:

.. doctest::

  >>> g = blaze.array([ 1, 2, 3], dshape='float32', storage=blaze.Storage('myarray.blz'))
  >>> g
  array([ 1.,  2.,  3.],
        dshape='3, float32')

You can use the persistent array as if it was an in-memory
array. However, it is persistent and it will survive your python
session. Later you can gain a reference to the array, even from a
different python session by name, using the `open` function:

.. doctest::

  >>> f = blaze.open(blaze.Storage('myarray.blz'))
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3, float32')

A persistent array is backed on non-volatile storage (currently, only
a filesystem is supported, but the list of supported storages may
increase in the future). That means that there are system resources
allocated to store that array, even when you exit your python
session.

A BLZ array can be enlarged anytime by using the `blaze.append()`
function, e.g.

.. doctest::

  >>> blaze.append(g, [4,5,6])
  >>> g
  array([ 1.,  2.,  3.,  4.,  5.,  6.],
        dshape='6, float32')

If you are done with the persistent array and want to free
its resources, you can just 'drop' it:

.. doctest::

  >>> blaze.drop(blaze.Storage('myarray.blz'))

After dropping a persistent array this way, any 'open' version you may
had of it will no longer be valid. You won't be able to reopen it
either. It is effectively deleted.


Working with CSV and JSON data files
------------------------------------

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


Working with HDF5 files
-----------------------

Blaze makes easy to work with HDF5 files via the included
`HDF5DataDescriptor`.  For the purposes of this tutorial we are going
to use some HDF5 files taken from the PO.DAAC project repository at JPL
(http://podaac.jpl.nasa.gov/).

Getting a Blaze object out of a dataset in the HDF5 file is easy, but
we need first a list of the datasets in the file.  For this, we are
going to use the standard HDF5 tool called `h5ls`:

.. doctest::

  In []: !h5ls test-daac.h5
  l3m_data                 Dataset {180, 360}
  palette                  Dataset {3, 256}

So, we have a couple of datasets named `l3m_data` and `palette`.
Let's open the latter with Blaze:

.. doctest::

  In []: store = blaze.Storage("test-daac.h5", format='hdf5')

  In []: palette = blaze.open(store, datapath="/palette")

As you see we needed first to create the usual `Storage` instance
where we are informing Blaze about the name and the format of the
file.  Then, we use `blaze.open()` with the `store` and the `datapath`
for the dataset inside the file that we wanted to open.  It is
important to understand that we just have a *handle* to the dataset,
but that we have not loaded any data in memory yet.  This handle
happens to be an actual Blaze ``Array`` object:

.. doctest::

  In []: type(palette)
  Out[]: blaze.objects.array.Array

which you can use as a lazy representation of the data on-disk, but
without actually reading the data.

Now, let's peek into the contents of the dataset:

.. doctest::

  In []: palette[1,1]
  Out[]: 
  array(255,
        dshape='uint8')

Or a slice:

.. doctest::

  In []: palette[1:3, 4:6]
  Out[]: 
  array([[255,   0],
         [255, 207]],
        dshape='2, 2, uint8')

