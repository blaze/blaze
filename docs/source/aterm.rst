=====
ATerm
=====

The Blaze python implementation of ATerm is designed to bring some of
the conveniences of pattern matching and rewriting capabilities of
languages like ML and Haskell into the Python world.

The ATerm representation of expression graphs forms the flexible
intermediate form that Blaze expressions can be manipulated with
in order to do optimization passes and execution dispatch.

Grammar
~~~~~~~

The ATerm spec is overseen by

.. _StrategoXT Project: http://strategoxt.org/Tools/ATermLibrary

The ``blaze.aterm`` is conforming implementation of the specification
with the exception that we exclude Binary Large Objects ``blob`` objects
are not present due to limitations in Python and limited usefullness in
Blaze.

The abstract grammar for ATerm is shown below.

::

    t : bt                 -- basic term
      | bt {ty,m1,...}     -- annotated term

    bt : C                 -- constant
       | C(t1,...,tn)      -- n-ary constructor
       | (t1,...,tn)       -- n-ary tuple
       | [t1,...,tn]       -- list
       | "ccc"             -- quoted string ( explicit double quotes )
       | int               -- integer
       | real              -- floating point number

Examples
~~~~~~~~

Example of ATerm expressions by category

:Integer: Integer terms (``1``)
:Real: Real terms (``2.718``)
:String: String terms (``"foo"``)
:Term: Variable terms (``x``)
:Application: Application of terms to other subterms (``f(x,y)``)
:List: A ordered list of subterms (``[1,2,3]``)
:Tuple: A ordered tuple of subterms (``(1,2,3)``)
:Placeholder: A placeholder for use in ``match`` and ``make`` commands (``<int>``)

**Simple Term**

.. code-block:: text

    x

.. code-block:: text

    foo

**Integers**

.. code-block:: text

    1

**Reals**

.. code-block:: text

    3.14159

**Strings**

.. code-block:: text

    "foo"

.. code-block:: text

    "foo \"bar\" fizz"

Strings must use double quotes, and can escape inner quotes using
backslash.

**Application**

.. code-block:: text

    f(x,y)

.. code-block:: text

    f(1,g(2,3))

.. code-block:: text

    Add(1,2)

In the above example the operand being applied is referred to as
the ``spine`` of the application with ``arguments``.

**Annotations**

.. code-block:: text

    x{prop}

.. code-block:: text

    x{prop1, prop2}

.. code-block:: text

    f(x{prop1}, y{prop2})

Annotations can contain any number of comma seperated aterm expression
with the restriction that annotations cannot themselves be annotated.

**Lists**

.. code-block:: text

   [1,2,3]

.. code-block:: text

   [3.141, 1, foo]

**Tuples**

.. code-block:: text

   (1, 2.718, 3.141)

.. code-block:: text

   (foo, bar)

Usage
~~~~~

::

    from blaze.aterm import parse, match, make

**Parser**

::

    >>> parse('x')
    aterm(term='x', annotation=None)

::

    >>> parse('f(x,y)')
    aappl(spine=aterm(term='f', annotation=None), args=[aterm(term='x', annotation=None), aterm(term='y', annotation=None)])

::

    >>> parse('x{prop1}')
    aterm(term=aterm(term='x', annotation=None), annotation=(aterm(term='prop1', annotation=None),))


**Matching**

Pattern matching is the action of determening whether a given aterm
expression conforms to pattern. It is similar in notion to regex. The
pattern may also contain placeholder objects which can be used to
deconstruct and pluck values of out aterms while rewriting.


:`<int>`: Matches int terms
:`<real>`: Matches real number terms
:`<str>`: Matches str terms
:`<term>`: Matches all terms
:`<placeholder>`: Matches all placeholder terms
:`<appl(...)>`: Matches all application to the specific arguments, ``(...)`` is not
    aterm syntax. See examples below.

The result a pattern match is a 2-tuple containing a boolean
indicating whether the match succeeded and a list containing the
capture valued.

::

    >>> match('x', 'x')
    (True, [])

::

    >>> match('x', 'y')
    (False, [])

::

    >>> match('f(<int>,<int>)', 'f(1,2)')
    (True, [aint(val=1), aint(val=2)])

::

    >>> match('<term>', 'x')
    (True, [aterm(term='x', annotation=None)])

::

    >>> match('f(<real>)', 'f(1)')
    (False, [])

::

    >>> match('Add(Succ(<int>), <term>)', 'Add(Succ(2), Succ(3))')
    (True,
     [aint(val=2),
      aappl(spine=aterm(term='Succ', annotation=None), args=[aint(val=3)])])

::

    >>> match('<appl(x,3)>', 'f(x,3)')
    (True, [aterm(term='f', annotation=None)])


For those coming from other languages, an analogy is uesfull. The
match operator in Prolog is written with `?`.

.. code-block:: prolog

    ?- x = x.
    true
    ?- x = y.
    false

Or often used to define functions which operate over pattern
matched variables collected on the LHS to free variables on the
RHS. For example in Prolog:

.. code-block:: prolog

    fact(0) => 0
    fact(n) => n*fact(n-1);

Or in ML:

.. code-block:: ocaml

    fun fact(1) = 1
      | fact(n) = n*fact(n-1);

Or in Haskell:

.. code-block:: haskell

    fact 0 = 1
    fact n = n * fact (n-1)


Motivating Examples
~~~~~~~~~~~~~~~~~~~

::

    aterm = namedtuple('aterm', ('term', 'annotation'))
    astr  = namedtuple('astr', ('val',))
    aint  = namedtuple('aint', ('val',))
    areal = namedtuple('areal', ('val',))
    aappl = namedtuple('aappl', ('spine', 'args'))
    atupl = namedtuple('atupl', ('args'))
    aplaceholder = namedtuple('aplaceholder', ('type','args'))

    # Lets try and match f(x,y) using pure Python
    def match_simple(term):
        if isinstance(term, appl):
            if isinstance(term.args[0], aterm):
                if isinstance(term.args[1], aterm):
                    if term.args[0].term == 'x':
                        if term.args[1].term == 'y':
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

    # Compared to
    match('f(x,y)', 'f(x,y)')
