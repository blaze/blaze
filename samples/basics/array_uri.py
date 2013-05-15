
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
