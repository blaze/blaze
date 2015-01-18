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

   df2 = pd.DataFrame({
      'name': 'Alice Bob Joe Bob Alice'.split(),
      'amount': np.random.rand(5) * 1000,
      'id': np.arange(5)
   })

   # put `df` into a Blaze Data object
   df = Data(df)
   df2 = Data(df2)


+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Computation     | Pandas                                                          | Blaze                                             |
+=================+=================================================================+===================================================+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|  Group By       |                                                                 |                                                   |
|                 |    df.groupby('name').amount.mean()                             |    by(df.name, amount=df.amount.mean())           |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
| Join            |                                                                 |                                                   |
|                 |    pd.merge(df, df2, on='name')                                 |    join(df, df2,                                  |
|                 |                                                                 |         on_left='name', on_right='name')          |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Selection       |    df[df.amount > 300]                                          |    df[df.amount > 300]                            |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Map             |    df.amount.map(lambda x: x + 1)                               |    df.amount.map(lambda x: x + 1,                 |
|                 |                                                                 |                  'float64')                       |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Relabel         |    df.rename(columns={'name': 'alias',                          |    df.relabel(name='alias',                       |
|                 |                       'amount': 'dollars'})                     |               amount='dollars')                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Drop duplicates |    df.drop_duplicates()                                         |    df.distinct()                                  |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Reductions      |    df.amount.mean()                                             |    df.amount.mean()                               |
|                 |    df.amount.value_counts()                                     |    df.amount.count_values()                       |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Projection      |    df[['id', 'amount']]                                         |    df[['id', 'amount']]                           |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+

Blaze can simplify and make more readable some common IO tasks that one would want to do with pandas. These examples make use of the `into <https://github.com/ContinuumIO/into>`_ project.


.. code-block:: python

   from into import into

+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Operation       | Pandas                                                          | Blaze                                             |
+=================+=================================================================+===================================================+
| Reading a       | .. code-block:: python                                          | .. code-block:: python                            |
| directory of    |                                                                 |                                                   |
| CSV files       |    df = pd.concat([pd.read_csv(filename)                        |    df = into(pd.DataFrame,                        |
|                 |                    for filename in                              |              'path/to/*.csv')                     |
|                 |                    glob.glob('path/to/*.csv')])                 |                                                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Reading in a    | .. code-block:: python                                          | .. code-block:: python                            |
| table from      |                                                                 |                                                   |
| a SQLite        |    import sqlalchemy as sa                                      |    df = into(pd.DataFrame,                        |
| database        |    engine = sa.create_engine('sqlite://db.db')                  |              'sqlite://db.db::t')                 |
|                 |    df = pd.read_sql('select * from t',                          |                                                   |
|                 |                     con=engine)                                 |                                                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
