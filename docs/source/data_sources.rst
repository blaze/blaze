===================================
Access Data from Different Sources
===================================

Blaze can access data from a variety of sources, as long as the format
has been implemented as a :ref:`Data Descriptor <data_descriptor>`.
Currently there are data descriptors for DyND, BLZ, HDF5, CSV and JSON
files, although the list is growing.  This section shows how to access
these files in a general way.

Working with BLZ storage
------------------------

BLZ is a storage solution that allows to keep your Blaze datasets
either in-memory or on-disk.

In-memory storage:

.. doctest::

  >>> import blz
  >>> ddesc = blaze.BLZ_DDesc(mode='w', bparams=blz.bparams(clevel=5))
  >>> arr = blaze.array([1, 2, 3], ddesc=ddesc)
  >>> print(arr)
  [1 2 3]

We can check that the array is backed by a the BLZ library by printing
the associated data descritor::

.. doctest::

  >>> arr.ddesc
  <blaze.datadescriptor.blz_data_descriptor.BLZ_DDesc object at 0x10d90dfd0>

an you can access the underlying BLZ object if you want too::

.. doctest::

  >>> arr.ddesc.blzarr
  barray((3,), int64)
    nbytes: 24; cbytes: 4.00 KB; ratio: 0.01
    bparams := bparams(clevel=5, shuffle=True, cname=blosclz)
  [1 2 3]

It is possible to force the type when creating the array. This
allows for a broader selection of types on construction:

.. doctest::

  >>> e = blaze.array([1, 2, 3], dshape='3 * float32', ddesc=ddesc)
  >>> e
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

Note that the dimensions in the datashape when creating from a
collection can be omitted. If that's the case, the dimensions will be
inferred. The following is thus equivalent:

.. doctest::

  >>> f = blaze.array([1, 2, 3], dshape='float32')
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

BLZ also supports arrays to be made persistent. This can be achieved
by adding the `path` parameter to the data descriptor:

.. doctest::

  >>> ddesc = blaze.BLZ_DDesc(path='myarray.blz', mode='w')
  >>> f = blaze.array([ 1, 2, 3], dshape='float32', ddesc=ddesc)
  >>> f
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

You can use the persistent array as if it was an in-memory
array. However, it is persistent and it will survive your python
session. Later you can gain a reference to the array, even from a
different python session by name, using the `from_*` functions:

.. doctest::

  >>> ddesc = blaze.BLZ_DDesc(path='myarray.blz', mode='a')
  >>> g = blaze.array(ddesc)
  >>> g
  array([ 1.,  2.,  3.],
        dshape='3 * float32')

A persistent array is backed on non-volatile data descriptor. That
means that there are system resources allocated to store that array,
even when you exit your python session.

A BLZ array can be enlarged anytime by using the `blaze.append()`
function, e.g.

.. doctest::

  >>> blaze.append(g, [4,5,6])
  >>> g
  array([ 1.,  2.,  3.,  4.,  5.,  6.],
        dshape='6 * float32')

If you are done with the persistent array and want to physically
remove its contents, you can just call the `remove()` method in the
associated data descriptor:

.. doctest::

  >>> g.ddesc.remove()

After removing a persistent array this way, any 'open' version you may
had of it will no longer be valid. You won't be able to reopen it
either. It is effectively deleted.


Working with CSV and JSON data files
------------------------------------

Let's suppose that we have a file named '/tmp/test.csv' that we want
to operate with from Blaze.  Blaze normally access data through
filesystem paths, so first, let's use the the `DDesc` class so as to
specify the file, as well as its schema:

.. doctest::

  >>> csv_schema = "{ f0: string, f1: string, f2: int16, f3: bool }"
  >>> ddesc = blaze.CSV_DDesc(path='test.csv', schema=csv_schema)

For JSON files it is just a matter of replacing the CSV data
descriptor by the JSON one:

.. doctest::

  >>> json_schema = "var * int8"
  >>> ddesc = blaze.CSV_DDesc(path='test.json', schema=json_schema)

Now, for actually accessing the data in the file we need to create a
Blaze array based on the descriptor:

.. doctest::

  >>> arr = blaze.array(ddesc)

As we see, the `array` constructor only needs you to pass the data
descriptor for your dataset and you are done.  If we want to have a
look at the contents, then just print the Blaze array:

.. doctest::

  >>> arr.ddesc.dynd_arr()  # workaround for flaky blaze print function
  nd.array([["k1", "v1", 1, false], ["k2", "v2", 2, true], ["k3", "v3", 3, false]], type="var * {f0 : string, f1 : string, f2 : int16, f3 : bool}")


Working with HDF5 files
-----------------------

Blaze makes easy to work with HDF5 files via the included
`HDF5DataDescriptor`.  For the purposes of this tutorial we are going
to use some HDF5 files taken from the PO.DAAC project repository at
JPL (http://podaac.jpl.nasa.gov/).

Getting a Blaze object out of a dataset in the HDF5 file is easy, but
we need first a list of the datasets in the file.  For this, we are
going to use the standard HDF5 tool called `h5ls`:

.. doctest::

  >>> !h5ls test-daac.h5
  l3m_data                 Dataset {180, 360}
  palette                  Dataset {3, 256}

So, we have a couple of datasets named `l3m_data` and `palette`.
Let's open the latter with Blaze:

.. doctest::

  >>> ddesc = blaze.HDF5_DDesc("test-daac.h5", datapath="/palette")
  >>> palette = blaze.array(ddesc)

As you see we needed first to create the usual `DDesc` instance where
we are informing Blaze about the name and the format of the file.
Then, we use `blaze.array()` with the data descriptor for actually
opening the dataset.  It is important to understand that we just have
a *handle* to the dataset, but that we have not loaded any data in
memory yet.  This handle happens to be an actual Blaze ``Array``
object:

.. doctest::

  >>> type(palette)
  >>> blaze.objects.array.Array

which you can use as a lazy representation of the data on-disk, but
without actually reading the data.

Now, let's peek into the contents of the dataset:

.. doctest::

  >>> palette[1,1]
  array(255,
        dshape='uint8')

Or a slice:

.. doctest::

  >>> palette[1:3, 4:6]
  array([[255,   0],
         [255, 207]],
        dshape='2 * 2 * uint8')

