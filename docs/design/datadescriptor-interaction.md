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

As we continue developing Data Descriptors this becomes a pretty serious
asset.  Consider for example loading data into a SQL database or extracting
data from HDFS.  These move beyond "just convenient" for novice users.

We may also use this internally.  It may make sense to move a csv file to HDF5
temporarily for a large computation, and then move it back.  For fancy string
representation or plotting we may want to move the head of an Array/Table to a
`Pandas_DDesc` and then use the work that that community has done in
formatting.


What needs to be done
---------------------

1.  Methods for bulk import/export

2.  Make creating empty DDescs natural.  Currently `CSV_DDesc(..., mode='w')`
    raises an error

3.  Create a `transfer` function (or `copy`, `dump`, ...)
    By default this can depend on

        for chunk in source.iterchunks:
            dest.extend(chunk)  # or append(chunk)

    But I think that we'll want to special case this for certain
    source/destination type pairs.  For example for CSV->CSV we can probably
    just perform a copy.  SQL->SQL is similar.


Extra things to do with Data Descriptors
----------------------------------------

1.  This would also be a good opportunity to test the DataDescriptors more
    rigorously


Questions
---------

We should think about how we communicate between these various descriptors.  My
understanding is that currently everyone can produce/consume DyND backed
descriptors.  It this true?  Is this our central point?

What should the "append many rows to collection" operation be called.  Options:

*   append
*   extend
*   appendmany

We need to have a deletion/clear/drop operation.  What should we call it?

Do we care about pairwise transfer relationships (like SQL->SQL)?  If so how do
we register these? (I have my
[solution](http://github.com/mrocklin/multipledispatch), but it's still beta)
