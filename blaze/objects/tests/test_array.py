from blaze import array
import blaze

def test_iter_1d():
    a = array([1, 2, 3])
    assert list(a) == [1, 2, 3]


def test_iter_nd():
    a = array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    assert list(list(a)[0]) == [1, 2, 3]

    assert blaze.all(list(a)[0] == array([1, 2, 3]))
    assert blaze.all(list(a)[1] == array([4, 5, 6]))
