'''Sample script showing off array basic operations'''

import blaze


def make_test_array(datashape):
    """TODO: Use something more interesting"""
    return blaze.zeros(datashape)

def test_operations(datashape):
    a = make_test_array(datashape)
    b = make_test_array(datashape)
    print ('a:')
    print (a)
    print ('b:')
    print (b)
    print ('a+b')
    print (a+b)
    print ('a-b')
    print (a-b)
    print ('a*b')
    print (a*b)
    print ('a/b')
    print (a/b)

if __name__ == '__main__':
    test_operations('10, float32')
    test_operations('10, int32')
    test_operations('10, 10, float32')
    test_operations('1000, 30, 25, float64')
    
