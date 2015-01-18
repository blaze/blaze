Pandas to Blaze
===============

This page maps pandas constructs to blaze constructs.


Imports and Construction
------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from blaze import Data, by, join

   # construct a DataFrame
   df = pd.DataFrame({
      'name': 'Alice Bob Joe Bob'.split(),
      'amount': np.random.rand(4) * 1000,
      'id': np.arange(4)
   })

   # put `df` into a Blaze Data object
   t = Data(df)


=============== ============================================================= ===========================================
**Computation** **Pandas**                                                    **Blaze**
--------------- ------------------------------------------------------------- -------------------------------------------
Group By        ``df.groupby('name').amount.mean()``                          ``by(t.name, amount=t.amount.mean())``
Join            ``pd.merge(df, df2, on='name')``                              ``join(s, t, on_left='name', on_right='name')``
Selection       ``df[df.amount > 300]``                                       ``t[t.amount > 300]``
Map             ``df.amount.map(lambda x: x + 1)``                            ``t.amount.map(lambda x: x + 1, 'float64')``
Relabel         ``df.rename(columns={'name': 'alias', 'amount': 'dollars'})`` ``t.relabel(name='alias', amount='dollars')``
Drop duplicates ``df.drop_duplicates()``                                      ``t.distinct()``
Reductions      ``df.amount.mean()``                                          ``t.amount.mean()``
Projection      ``df[['id', 'amount']]``                                      ``t[['id', 'amount']]``
=============== ============================================================= ===========================================
