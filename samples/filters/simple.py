'''Sample script showing off some simple filter operations'''

from __future__ import absolute_import, division, print_function

import blaze


def make_array(path):
    ddesc = blaze.BLZ_DDesc(path, mode='w')
    arr = blaze.array([(i, i*2.) for i in range(100)],
                      'var * {myint: int32, myflt: float64}',
                      ddesc=ddesc)
    return arr


if __name__ == '__main__':
    # Create a persitent array on disk
    arr = make_array("test.blz")
    # Do the query
    res = [i for i in arr.where('(myint < 10) & (myflt > 8)')]
    print("Results of the filter:", res)
    # Remove the persitent array
    arr.ddesc.remove()

