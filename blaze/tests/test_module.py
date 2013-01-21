import os

import blaze
from blaze.module.parser import mopen, mread

#------------------------------------------------------------------------
# Dummy Modules
#------------------------------------------------------------------------

dummy = """
interface Dummy t:
    fun foo :: (t,a) -> a
    fun bar :: (t,a,b) -> (a,b)
    fun baz :: t -> t
    fun pop :: t -> !dirty t
"""

dummy_expected = \
{'interfaces': {'Dummy': {'defs': {'bar': {'sig': {'cod': ['a', 'b'], 'dom': ['t', ['a', 'b']]}},
          'baz': {'sig': {'cod': 't', 'dom': 't'}},
          'foo': {'sig': {'cod': 'a', 'dom': ['t', 'a']}},
          'pop': {'sig': {'cod': ('!', 'dirty'), 'dom': 't'}}},
 'name': 'Dummy',
 'params': ('t',)}}}

#------------------------------------------------------------------------

arith = """
interface Arith t:
    fun add      :: (t,t) -> t
    fun multiply :: (t,t) -> t
    fun subtract :: (t,t) -> t
    fun divide   :: (t,t) -> t
    fun mod      :: (t,t) -> t
    fun power    :: (t,t) -> t
"""

arith_expected = \
{'interfaces': {'Arith': {'defs': {'add': {'sig': {'cod': 't', 'dom': ['t', 't']}},
          'divide': {'sig': {'cod': 't', 'dom': ['t', 't']}},
          'mod': {'sig': {'cod': 't', 'dom': ['t', 't']}},
          'multiply': {'sig': {'cod': 't', 'dom': ['t', 't']}},
          'power': {'sig': {'cod': 't', 'dom': ['t', 't']}},
          'subtract': {'sig': {'cod': 't', 'dom': ['t', 't']}}},
 'name': 'Arith',
 'params': ('t',)}}}

#------------------------------------------------------------------------

def test_dummy():
    print mread(dummy)

def test_arith():
    print mread(arith)

#------------------------------------------------------------------------
# Blaze Core
#------------------------------------------------------------------------

def test_blaze_core():
    path = os.path.dirname(blaze.__file__)
    blazecore = os.path.join(path, 'module', 'blaze.mod')

    mopen(blazecore)
