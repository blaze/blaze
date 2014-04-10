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

## Rolling Sum

## Rolling Mean

## Rolling Min/Max

See also

* [Elementwise Reduction UFuncs](elwise-reduction-ufuncs.md)
