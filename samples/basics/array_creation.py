'''Sample module showing the creation of blaze arrays'''
from __future__ import print_function

import blaze

def print_section(a_string, spacing=2, underline='='):
    print ('%s%s\n%s' % ('\n'*spacing,
                         a_string,
                         underline*len(a_string)))


print_section('building basic arrays')
# It is possible to build arrays from python lists
a = blaze.array([ 2, 3, 4 ])

# Arrays can be printed
print (a)

# The array will have a datashape. A datashape is a combination of the
# shape and dtype concept found in numpy. Note that when creating from
# a Python iterable, a datashape will be inferred.
print (a.dshape)

b = blaze.array([1.2, 3.5, 5.1])
print (b)
print (b.dshape)

# Arrays can be bi-dimensional
print_section('going 2d', spacing=1, underline='-')
c = blaze.array([ [1, 2], [3, 4] ]) 
print (c)
print (c.dshape)

# or as many dimensions as you like
print_section('going 3d', spacing=1, underline='-')
d = blaze.array([ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ])
print (d)
print (d.dshape)

# --------------------------------------------------------------------

print_section ('building compressed in-memory arrays')

# A compressed array (backed by BLZ):
blz = blaze.array([1,2,3], caps={'compress': True})
print (blz)

# --------------------------------------------------------------------

print_section('Explicit types in construction')
# It is possible to force a type in a given array. This allows a
# broader selection of types on construction.
e =  blaze.array([ 1, 2, 3], dshape='3, float32') 
print (e)

# Note that the dimensions in the datashape when creating from a
# collection can be omitted. If that's the case, the dimensions will
# be inferred. The following is thus equivalent:

f = blaze.array([ 1, 2, 3], dshape='float32')
print (f)

# --------------------------------------------------------------------

print_section('Alternative  constructors')

# Arrays can be created to be all zeros:
g = blaze.zeros('10, 10, int32')
print (g)

# All ones
h = blaze.ones('10, 10, float64')
print (h)

# --------------------------------------------------------------------

print_section('Indexing')

print ('starting with a 4d array')
array4d = blaze.ones('10,10,10,10, float32')
print (array4d)

print ('index once')
print (array4d[3])
print ('index twice')
print (array4d[3,2])
print ('index thrice')
print (array4d[3,2,4])
print ('index 4 times')
print (array4d[3,2,4,1])
