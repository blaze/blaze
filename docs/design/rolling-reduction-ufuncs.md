# Rolling Reduction UFuncs

## General Discussion

A rolling reduction is a function whose domain is
intervals of the source array as follows:

```
def rolling_reduce(op, a, winsize):
    """
    Apply `op` as a rolling reduction, with trailing
    windows.
    """
    # Start with NaN until we reach the window size
    res = [float('nan')] * (winsize - 1)
    for i in range(winsize - 1, len(a)):
        win = a[i - winsize + 1: i + 1]
        res.append(op(win))
    return res

def rolling_max(a, winsize):
    from functools import reduce # in Python 3
    def arrmax(win):
        return reduce(max, win)
    return rolling_reduce(arrmax, a, winsize)

>>> rolling_max([1, 3, 7, float('nan'), 6, 2, 7, float('inf')], 3)
[nan, nan, 7, 7, 7, nan, 7, inf]
```

Note that this is actually incorrect, because NaN is
not orderable.

```
>>> reduce(max, [float('nan'), 2, 1])
nan

>>> reduce(max, [1, float('nan'), 2])
2
```

Here's one way to fix this:

```
def rolling_max(a, winsize):
    from functools import reduce # in Python 3
    def arrmax(win):
        if all(x == x for x in win):
            return reduce(max, win)
        else:
            return float('nan')
    return rolling_reduce(arrmax, a, winsize)

>>> rolling_max([1, 3, 7, float('nan'), 6, 2, 7, float('inf')], 3)
[nan, nan, 7, nan, nan, nan, 7, inf]
```

Pandas adds some flexibility to handling of NaN values
by adding a 'minp'/'min_periods' parameter, which
is a minimum number of non-NaN values to require for
getting a finite result. Code for it might look
like this:

```
def rolling_max(a, winsize, minp):
    from functools import reduce # in Python 3
    def arrmax(win):
        present = sum(x == x for x in win)
        if present >= minp:
            return reduce(max, (x for x in win if x == x))
    return rolling_reduce(arrmax, a, winsize)

>>> rolling_max([1, 3, 7, float('nan'), 6, 2, 7, float('inf')], 3, 2)
[nan, nan, 7, 7, 7, 6, 7, inf]
```

## References

* https://en.wikipedia.org/wiki/Moving_average
* http://pandas.pydata.org/pandas-docs/stable/computation.html#moving-rolling-statistics-moments

## Pandas Implementation

The core of Pandas' rolling sum and rolling mean are
here in the codebase:

https://github.com/pydata/pandas/blob/master/pandas/algos.pyx#L840

Characteristics of the implementation include:

* Outputs NaNs until the window size is reached.
* Keeps track of the number of non-NaNs encountered
* Has parameter `minp` controlling the number of
  non-NaN values required to return a finite value.
* 
* The data is preprocessed with a 'kill_inf'
  option converting infinities to NaN.
  * This has implications, such as rolling_max
    incorrectly giving NaN instead of +inf
    where +inf appears, or NaN instead of the
    valid finite values where -inf appears.

## Miscellaneous Notes

* In science and engineering, a centred window is
  common, but in financial applications a trailing
  window is used. I assume the latter is because in
  you don't want the statistic to include information
  from the future in a time series analysis.

# Efficient Implementations

For a number of rolling reductions, it's possible to
create implementations that do not have a ``O(winsize)``
factor in them. Here we describe a few of these.

## Rolling Sum

We are computing the sequence

```
S_i = a_(i-k+1) + ... + a_i
```

and with a simple substraction, we get the ``O(1)`` update
formula

```
S_(i+1) = S_i + a_(i+1) - a_(i-k+1)
```

Both NaN and inf values cause some small computations,
because they don't satisfy ``(x - inf) + inf == x``.
Correct handling of these requires counting instances of
inf and NaN.

Additionally, this update formula will not produce the
exact same answer as recomputing the sum for each window,
because of floating point inaccuracy. Using a higher
precision float for the accumulator than the array values
is one way to mitigate this a bit. See the next rolling
mean section for a particular example of this which
was encountered in Pandas.

## Rolling Mean

Mean is a slight modification to sum, where the count of
non-NaN elements is used as the divisor.

In the Pandas implementation, an additional tweak is done,
where the number of negative values is tracked
separately, and the value is clamped to zero if the value
is negative when no negative values are in the sum. This
was introduced to deal with https://github.com/pydata/pandas/issues/2527,
where a sum of all non-negative numbers was producing a
negative value, causing an "impossible" square root failure.

Taking a look at the code for the particular failing example,
it looks like doing the subtract before the add would fix the
issue, though likely not fix it in general. Here's how
changing the order of computation affects the result in that
issue:

```
>>> ((0.00012456 + 0.0003) - 0.00012456) - 0.0003
-5.421010862427522e-20

>>> ((0.00012456 - 0.00012456) + 0.0003) - 0.0003
0.0
```

This issue affects both rolling sum and rolling
mean, however we deal with it should be done the same in both.

## Rolling Min/Max

* http://people.cs.uct.ac.za/~ksmith/articles/sliding_window_minimum.html
* http://richardhartersworld.com/cri/2001/slidingmin.html

This algorithm is sometimes known as Ascending Minima.

The trick with rolling min/max is to keep track of the set of
all values which might affect a future result. Let's take
a look at what these sets look like for the example data we
used in the general discussion.

```
Value    Set
-----    ---
1
3
7        {7}
nan      {nan}
6        {nan, 6}
2        {nan, 6, 2}
7        {7}
inf      {inf}
3        {inf, 3}
```

Since NaN is not part of the totally ordered set, it needs to
be tracked separately, but the rest of the values including
inf can be handled uniformly. We need a bounded deque, which
could be C++'s std::deque, or a circular buffer like boost has
(http://www.boost.org/doc/libs/1_55_0/doc/html/circular_buffer.html).
We store pairs ``(val, index)`` in this buffer, in sorted
order. When processing, we always discard pairs whose
``index`` is too old, or which can no longer affect a later
value because the new source value is dominant.

Let's work out the values seen in the buffer:

```
Index Value    NaN?  Buffer
----- -----    ----  ------
0     1              [(1, 0)]
1     3              [(3, 1)]
2     7              [(7, 2)]
3     nan      3     []
4     6        3     [(6, 4)]
5     2        3     [(6, 4), (2, 5)]
6     7              [(7, 6)]
7     inf            [(inf, 7)]
8     3              [(inf, 7), (3, 8)]
```

And some Python code:

```
def rolling_max(a, winsize, minp):
    pass # WIP
```

See also

* [Elementwise Reduction UFuncs](elwise-reduction-ufuncs.md)
