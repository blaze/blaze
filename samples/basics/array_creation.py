'''Sample module showing the creation of blaze arrays'''

import blaze

print ('building basic arrays')
# It is possible to build arrays from python lists
a = blaze.array([ 2, 3, 4 ])

# Arrays can be printed
print (a)

# The array will have a datashape. A datashape is a combination of the
# shape and dtype concept found in numpy. Note that when creating from
# a Python iterable, a datashape will be inferred.
print a.datashape

b = blaze.array([1.2, 3.5, 5.1])
print b.datashape

# Arrays can be bi-dimensional
print ('going 2d')
c = blaze.array([ [1, 2], [3, 4] ]) 
print (c)
print (c.datashape)

# or as many dimensions as you like
print ('3d')
d = blaze.array([ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ])

print (d.datashape)

# --------------------------------------------------------------------

print ('Explicit types in construction')
# It is possible to force a type in a given array. This allows a
# broader selection of types on construction.
e =  blaze.array([ 1, 2, 3], dshape='3, float32') 
print (e)

# Note that the dimensions in the datashape when creating from a
# collection can be omitted. If that's the case, the dimensions will
# be inferred. The following is thus equivalent:

f = blaze.array([ 1, 2, 3], dshape='float32')


# --------------------------------------------------------------------

print ('Alternative  constructors')

# Arrays can be created to be all zeros:
g = blaze.zeros('10, 10, int32')
print (g)

# All ones
h = blaze.ones('10, 10, float64')
