'''Sample script showing the usage of the array uri related API'''

# This sample aims to provide the gist of Blaze's uri API.  This part
# of the API allows the usage of arrays that are 'stored' in a given
# uri, or associating the array with an uri.
#
# Using uris allow locating arrays on external storage, allowing for
# persistence and sharing.

# The API consists of:
#
# open - allows creating a blaze array from an uri, working with the
# data directly in the data source.
#
# create - allows creating a blaze array in a given uri.
#
# create_fromiter - allows creating a blaze array in a given uri,
# populating it with data from the iterator.
#
# load - creates an in-memory copy of a blaze array from an array in
# the specified uri
#
# save - takes a blaze array and saves it into the specified uri.
# This involves copying.
#
# drop - remove an array 

from __future__ import print_function

from blaze

def print_section(a_string, spacing=2, underline='='):
    print ('%s%s\n%s' % ('\n'*spacing,
                         a_string,
                         underline*len(a_string)))

print_section('start with basic array')
a = blaze.array([ 2, 3, 4, 5, 6])

print (a)

# ----------------------------------------------------------------------

print_section('Basic save/load')
sample_uri='blz://load_save'

# save it in a uri
blaze.save(a, sample_uri)

# load it...
b = blaze.load(sample_uri)

print(b)

# ----------------------------------------------------------------------

print_section('using an array with open')

# It is possible to open an existing array
c = blaze.open(sample_uri)

print(c)

# whilst it may seem that open and load are equivalent, they are
# actually different. When you load an array, you make an in-memory
# copy of the array. However, when using open you are using the data
# backed in the disk without a copy. Once loaded, operating in an
# in-memory array may be faster, however, the open will be faster than
# load. A loaded blaze array may incur into higher memory usage, while
# in the case of an "open" one it will just keep some of the disk data
# buffered.

# ----------------------------------------------------------------------

print_section('droping an array')

# we can get rid of an array with drop
blaze.drop(sample_uri)

# ----------------------------------------------------------------------

print_section('creating uri arrays')

# Appart from creating arrays using the save function, it is possible
# to create empty arrays and populate them with appends. This is a
# great way to build huge arrays without needing huge amounts of
# memory. After that it is possible to just 'open' the uri, so data
# can be accessed in an out-of-core way.

# When creating an array datashape must be provided that specifies the
# measure and all but the outer dimension. The outer dimension will be
# '0', and will grow as data gets appended.

create_uri = 'blz://create_sample'
create_dshape = 'uint32'
created_array = blaze.create(create_uri, create_dshape)

# created_array should now have a dshape of '0, uint32'
print(created_array.dshape)
# and it should print as an empty array []
print(created_array)

def fib(n):
    a, b = 0, 1
    while n != 0:
        yield a
        a, b = b, a+b
        n -= 1

for i in fib(300):
    created_array.append(i)

# The created array should now have a dshape of '300, uint32'
print(created_array.dshape)

# And it should contain a nice fibonacci sequence
print(created_array)

del created_array
drop(create_uri)

# ----------------------------------------------------------------------

print_section('creating uri arrays from iterators')

# it is possible to create an uri array directly from an iterator. For
# example, we could achieve the same result as in the previous section
# with just the following:
created_array = create_fromiter(create_uri, create_dshape, fib(300))
print(created_array.dshape)
print(created_array)
