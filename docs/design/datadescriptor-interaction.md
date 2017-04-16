DataDescriptor Interaction
--------------------------

We propose functionality to support bulk transfer of information between Data
Descriptors.

```Python
schema = ...
csv = CSV_DDesc('my-file.csv', schema=schema)
json = CSV_DDesc('my-file.json', 'w', schema=schema)

transfer(csv, json)
```

At first this seems only very convenient.  Through our Data Descriptors we
happen to have organized hooks to a number of common formats.  Providing simple
transfer between them might be of value on it's own, blaze arrays be damned.

As we continue developing Data Descriptors though this becomes a more serious
asset.  Consider for example loading data into a SQL database or extracting
data from HDFS.  These move beyond "just convenient" for novice users.

We may use this internally.  It may make sense to move a csv file to HDF5
temporarily for a large computation, and then move it back.  For fancy string
representation or plotting we may want to move the head of an Array/Table to a
`Pandas_DDesc` and then use the work that that community has done in
formatting.

*Intuitive and efficient movement between data representations can improve
workflow.*


What needs to be done
---------------------

1.  Methods for bulk import/export

2.  Make creating empty DDescs natural.  Currently `CSV_DDesc(..., mode='w')`
    raises an error

3.  Create a `transfer` function (or `copy`, `dump`, ...)
    By default this can depend on

        for chunk in source.iterchunks:
            dest.extend(chunk)  # or append(chunk)

    But I we may eventually want to special case this for certain
    source/destination type pairs.  For example for CSV->CSV we can probably
    just perform a copy (if dialects are equivalent).  Similarly SQL->SQL
    doesn't need to flow through the local machine, it can be done purely on
    the database server.


Extra things to do with Data Descriptors
----------------------------------------

1.  This would also be a good opportunity to test the DataDescriptors more
    rigorously


Proposal
--------

Data Descriptors have two main input/output pipes

### Output

*   `__iter__` - a lazy iterator of items/rows out of the data.  This is user
    facing and so returns user friendly representations, often python objects
    or DyND arrays.
*   `iterchunks` - a lazy iterator of largeish chunks of data.  These chunks
    are DataDescriptors (often `DyND_DDesc`), fit in memory, and are sizable
    enough to enable optimizations.  These are not user friendly but do come
    with datashape information.

### Input

We have two choice pairs

*   User friendly types / DataShape embued types
*   One at a time / sequence of data

So for example the user might reasonably want to append the single Python
object

    {'name': 'Alice', 'amount': 100}

Or the user might want to add many such objects at once.  These two choices
are common and have different performance considerations.

Additionally we will internally want to move data around.  We want to move
chunks of datashape imbued data around.

Current working names

*   `append` - add a single Python object to collection
*   `extend` - add a sequence of Python objects to collection
*   `append_chunk` - add a single `DDesc` to collection
*   `extend_chunks` - add a sequence of `DDesc`s to collection

At least `extend`, `extend_chunks` and possibly `append` are useful.  We should
decide on how verbose of an interface we want here.


Storage Types
-------------

Currently in `datadescriptor` we support at least the following:

*   csv
*   json
*   DyND
*   HDF5
*   BLZ

There are also various other more exotic systems.  `io/sql/` has a SQL backed
data descriptor.  `catalog` handles data on disk to a certain extent.
Presumably we'll eventually want to interact with data on `HDFS`.


Storage and Computation
-----------------------

We don't yet have well defined boundaries between `compute` and
`datadescriptor`.  In particular it's not clear what `compute` can expect
`ddesc` to be capable of.

As an example consider Francesc's recent work hooking an array operation
`where` to BLZ and HDFS data descriptors.  He was able to hook into PyTables'
selection routines to power the HDFS solution.  In this case the library that
we use to manage the storage, PyTables, also has considerable computational
strength and so we win in this case.  The same is true for SQL.  What is SQL, a
storage mechanism or a computational backend?

This distinction is strongest if you compare `csv` to `SQL`.  We're going to be
tempted to put computational functionality on `SQL_DDesc` that we can't expect
`CSV_DDesc` to efficiently support.


### Proposal

*I'm not confident in these ideas, please challenge*

Data Descriptors should restrict themselves to very simple storage and
retrieval operations, even when the backends are capable of more.  Any other
strategy will result in a mish-mash of methods and no clear interface.

This restrictive approach is ok, we'll leverage the more computational backends
separately in the computation layer.  Lets get a solid and useful data layer
and then work our way up.

So what operations *should* we expect?

Efficient selection and insertion of chunks of rows of data

    # Pull data out of data descriptor
    my_ddesc.iterchunks()

    # Shove data into data descriptor
    my_ddesc.extend_chunks(source_of_chunks)

Efficient (hopefully!) selection of a set of rows of data, e.g.

    index = [1, 5, 6, 7, 12, 20]  # rows about which we care
    values = my_ddesc[index]      # DyND_DDesc of those rows

What else is necessary?  I think that we can make a reasonable default
computational engine with only these operations.  I think that we can expect
any data descriptors to implement all of these operations.


### Example

I came to this way of thinking as I was working on getting `HDF5_DDesc` to
robustly support my input/output interface above.  There is a tension between
using two Python libraries for HDF5 interaction, `h5py` and `PyTables`.

The `h5py` library offers a more direct/raw mapping of HDF5 functionality while
PyTables wraps HDF5 with a bit of logic and offers very fast and useful
operations on top of HDF5 (efficient selection, connection to `numexpr`, ...).

Because I generally prefer simpler technologies my first reaction was to go for
`h5py`, I found its model more in line with my understanding of HDF5.  However
multiple people have pointed out that the computational abilities of PyTables
really *are* impressive and very much in line with our applications (not to
mention we have an expert in house :) ).

H5Py is more like CSV while PyTables is really more like SQL.

My current view is that if the two libraries interoperate well (e.g. PyTables
doesn't depend on custom data that it injects into the HDF5 files) then we use
H5Py in the data layer and use PyTables as a computational backend.  I haven't
yet spoken to Francesc about this though and given his experience on the
topic I hope that he can contribute a lot to develop these questions.

Anyway, I think that the distinction here is interesting and reflects a broader
conflict that we're working through in Blaze.
