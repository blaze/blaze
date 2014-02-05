Fusion
======

This document addresses the problem of deforestation, or fusion, in blaze. We
shortly explain why fusion is important, after which we detail what the
current design allows, and what it's shortcoming are, with possible solutions.

Introduction
------------
Fusion is an optimization that eliminates temporary arrays, lists, trees, etc.
For instance, in the element-wise expression `a + b * c`, we want to loop
over all elements of a, b and c simulteneously, while writing to some output
array. We don't want to allocate a temporary in which we evaluate `b * c` only
to subsequently read the data back in to evaluate `a + temp`. This causes
unneccessary traffic to load data from memory into cache, and from cache into
registers (or even from disk into memory first).

Current State
-------------
We currently fuse expressions at their most inner levels. For instance, we
have a notion of "element-wise" functions which operate on scalars. These
operations can always be fused, such that all fused scalar functions can
be applied in succession:

```python
    for i in range(shape[0]):
        for j in range(shape[1]):
            t = b[i, j] * c[i, j]
            out[i, j] = a[i, j] + t
```

We support the notion of generalized ufuncs, which operate instead over
N dimensions, leaving broadcasting implicit in outer dimensions. For instance:

```python
def filter1d(array, conditions):
    return [item for item, cond in bzip(array, conditions) if cond]
```

Shortcomings
------------
A shortcoming of this design is that it is unclear how the computations inside
the generalized ufuncs in this system can be fused together without an
optimizing compiler. A problem is the mixture of languages and systems that
need to cooperate. For instance, in the following exmaple:

```python
sum(filter1d(arr, conds))
```

how can we avoid building all the intermediate filter lists or arrays at the
inner level? It seems the ability to fuse this expression depends on both
the nature of filter and sum: filter does not need to process all data in order
to yield an element or chunk, and sum doesn't need all data available in order
to perform any work.

It makes sense to allow pushing or pulling of data at a more fine-grained level.
In an optimizing compiler, this could be just elements. However, due to the overhead
of dynamic dispatch inherent to a static system implementing this, we need
a tradeoff between dynamic dispatch overhead and the cost of temporaries.
Instead, we can push around blocks of data. This can be done in several
ways:

    * tasks or coroutines
    * continuation passing style (CPS)
    * iterators

Tasks
-----
Write producers as cooperatively scheduled green-threads that push and pull
blocks of data. This correspconds directly to our expression graph:

```python
def filter(inputs, outputs):
    for (cond_block, item_block) in inputs.receive_all():
        result_block = [
        for cond, item in zip(cond_block, item_block):
            if cond:
                result_block.append(item)

        outputs.send(result_block)

def sum(inputs, outputs):
    sum = 0
    for block in inputs.receive_all():
        for item in block:
            sum += x

    outputs.send(sum)
```

This system terminates when all tasks have died. We leave implicit here the
closing of the channels that will allow `receive_all` to stop reading.

This system is elegant because it is modular and easy to extend with new
functions that cooperate under the same conditions. If someone along the chain
does need all input data, that task can simply perform buffering.

Continuation Passing Style
--------------------------
Instead of returning chunks, we can instead communicate them through
continuations. We can pass around blocks by default, and we can
support convenience code to perform buffering where necessary. On the other
hand you can annotate implementions with `element`, `block` or `full` buffering,
which automatically handles buffering/unbuffering, making the system more
powerful (it would be relatively straightforward for an optimizing compiler
to eliminate the dispatch on continuations).

Below we demonstrate how to make it work using just scalars. Instead, you can
imagine blocks being pushed through.

```python
def add(a, b, cont):
    cont(a + b)

def mul(a, b, cont):
    cont(a * b)

def expr(a, b, c, d, cont):
    mul(a, b, lambda ab: mul(c, d, lambda cd: add(ab, cd, cont)))

print((a * b) + (c * d))
expr(a, b, c, d, print)
```

Iterators
---------
Perhaps the easiest yet modular way to express the problem is in terms
of multi-dimensional iterators, where each iterator returns successive blocks
that are computed over:

```python
def filter(input, conds):
    for item, cond in zip(input, conds):
        if cond:
            yield item

def sum(input):
    result = input[0]
    for item in input[1:]:
        result += item
    return result
```

Multiple dimensions are simply expressed through nested iterators.

Iterators can be implemented as part of ckernels, where each dimension produces
an iterator of the dimension it wraps. The innermost kernel does some buffering
to accumulate results in blocks, which it returns. These iterators simply
wrap expressions:

```
template<typename Inputs..., typename Output, typename E>
class Iterator {
public:
    Output produce() {
        return expr.apply(iterators...);
    }
};

template<typename T>
class Add {
    T apply(T a, T b) {
        return a + b;
    }
};
```