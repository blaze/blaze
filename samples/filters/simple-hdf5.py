'''Sample script showing off some simple filter operations (HDF5 version)'''

from __future__ import absolute_import, division, print_function

import blaze


def make_array(path):
    ddesc = blaze.HDF5_DDesc(path, '/table', mode='w')
    arr = blaze.array([(i, i*2.) for i in range(100)],
                      'var * {myint: int32, myflt: float64}',
                      ddesc=ddesc)
    return arr


if __name__ == '__main__':
    # Create a persitent array on disk
    arr = make_array("test-filtering.h5")
    # Do the query
    res = arr.where('(myint < 10) & (myflt > 8)')
    # Print out some results
    print("Resulting array:", res)
    # Materialize the iterator in array and print it
    print("\nResults of the filter:\n", list(res))
    # Remove the persitent array
    arr.ddesc.remove()
