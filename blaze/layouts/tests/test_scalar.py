from blaze.layouts.scalar import Interval, Chart, vstack,\
    hstack, dstack

def test_multiple_charts():
    alpha = object()
    beta  = object()

    a = Interval(0,2)
    b = Interval(0,2)

    x = Chart([a,b], alpha)
    y = Chart([a,b], beta)

    # -------------
    s = hstack(x,y)
    # -------------

def test_vertical_stack():
    alpha = object()
    beta  = object()

    a = Interval(0,2)
    b = Interval(0,2)

    x = Chart([a,b], alpha)
    y = Chart([a,b], beta)

    # -------------
    s = vstack(x,y)
    # -------------

    block, coords = s[[3,1]]
    assert block is beta
    assert coords == [1,1]

    block, coords = s[[0,0]]
    assert block is alpha
    assert coords == [0,0]

    block, coords = s[[1,0]]
    assert block is alpha
    assert coords == [1,0]

    block, coords = s[[2,0]]
    assert block is beta
    assert coords == [0,0]

    block, coords = s[[2,1]]
    assert block is beta
    assert coords == [0,1]

def test_horizontal_stack():
    alpha = object()
    beta  = object()

    a = Interval(0,2)
    b = Interval(0,2)

    x = Chart([a,b], alpha)
    y = Chart([a,b], beta)

    # -------------
    s = hstack(x,y)
    # -------------

    block, coords = s[[0,0]]
    assert block is alpha
    assert coords == [0,0]

    block, coords = s[[0,1]]
    assert block is alpha
    assert coords == [0,1]

    block, coords = s[[0,2]]
    assert block is beta
    assert coords == [0,0]

    block, coords = s[[2,4]]
    assert block is beta
    assert coords == [2,2]

def test_third_axis():
    alpha = object()
    beta  = object()

    a = Interval(0,2)
    b = Interval(0,2)
    c = Interval(0,2)

    x = Chart([a,b,c], alpha)
    y = Chart([a,b,c], beta)

    # -------------
    s = dstack(x,y)
    # -------------

    block, coords = s[[0,0,0]]
    #assert block is alpha
    assert coords == [0,0,0]
