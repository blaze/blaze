Our overall goal is to focus on Blaze from an end user's (and new user's)
perspective.  We want to identify common usability issues, address them, and
help make blaze a more useful technology.

One way we want to improve Blaze is by opening it up and allowing it to work
better with Python as a whole.

To this end, we want to build a bridge between users' existing Python
knowledge and Blaze.  We can use this bridge to help new users better
understand Blaze expressions, and we can highlight the correspondences.

There is an interesting correspondence between Blaze expressions and Python
functions:

Blaze | Python
------|-------
expression | (pure) Python function
symbol     | unbound function arguments
bound symbol | function closure
`compute(expr, data)` | `func(**data)`

We can take this correspondence further, and we think that would be a good thing to do:

Python | Blaze
------|-------
function docstring | automatic blaze expression docstring, with ability for end users to 



Improve Blaze design to play nicely with rest of Python, highlight correspondences:
Blaze expressions ⇔ Python functions
Blaze symbols ⇔ unbound function arguments
Blaze bound symbols ⇔ closure / default argument values
(new) automatic blaze expr docstrings ⇔ function docstrings
(new) callable expressions ⇔ calling Python function
(new) composing blaze expressions ⇔ composing Python functions
(new) computational pipelines in Blaze: 
pipeline = comp(filter, grouper, selector, summarizer)
pipeline(data) => summarize(select(group(filter(data))))
Blaze free to compose and optimize expressions in pipeline.
Remove distinction between “interactive” and “non-interactive” interaction with Blaze.
Extremely confusing to new users.  Change terminology, documentation, API naming, etc.
Still keep bound and unbound symbols.
Expressions can be fully unbound, partially bound, or fully bound.  Have to have a fully bound expression to actually compute a result.
compute(expr, {mapping for remaining unbound symbols})
make all computation explicit, but still easily accessible.
No more implicit computation in `__repr__`.
`compute()` changed so that it actually computes, every time.
if we want compilation of blaze expression to SQLA query, call that something else (`compile()` maybe?).  Can be used as part of compute’s operation.
By default, compute returns results in default backend, odos to default backend if necessary.
all table-like things: default is Pandas; dask.dataframe if ooc.
all array-like things: default is NumPy; dask.array if ooc.
Overrideable behavior with kwarg
