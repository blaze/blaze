Pandas to Blaze
===============

This page maps pandas constructs to blaze constructs.

Imports and Construction
------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from blaze import data, by, join, merge, concat

   # construct a DataFrame
   df = pd.DataFrame({
      'name': ['Alice', 'Bob', 'Joe', 'Bob'],
      'amount': [100, 200, 300, 400],
      'id': [1, 2, 3, 4],
   })

   # put the `df` DataFrame into a Blaze Data object
   df = data(df)


+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Computation     | Pandas                                                          | Blaze                                             |
+=================+=================================================================+===================================================+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
| Column          |                                                                 |                                                   |
| Arithmetic      |    df.amount * 2                                                |    df.amount * 2                                  |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
| Multiple        |                                                                 |                                                   |
| Columns         |    df[['id', 'amount']]                                         |    df[['id', 'amount']]                           |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Selection       |    df[df.amount > 300]                                          |    df[df.amount > 300]                            |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
| Group By        |                                                                 |                                                   |
|                 |    df.groupby('name').amount.mean()                             |    by(df.name, amount=df.amount.mean())           |
|                 |    df.groupby(['name', 'id']).amount.mean()                     |    by(merge(df.name, df.id),                      |
|                 |                                                                 |       amount=df.amount.mean())                    |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
| Join            |                                                                 |                                                   |
|                 |    pd.merge(df, df2, on='name')                                 |    join(df, df2, 'name')                          |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Map             |    df.amount.map(lambda x: x + 1)                               |    df.amount.map(lambda x: x + 1,                 |
|                 |                                                                 |                  'int64')                         |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Relabel Columns |    df.rename(columns={'name': 'alias',                          |    df.relabel(name='alias',                       |
|                 |                       'amount': 'dollars'})                     |               amount='dollars')                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Drop duplicates |    df.drop_duplicates()                                         |    df.distinct()                                  |
|                 |    df.name.drop_duplicates()                                    |    df.name.distinct()                             |
|                 +-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block::python                                           | .. code-block::python                             |
|                 |                                                                 |                                                   |
|                 |    df.drop_duplicates(subset=('name', 'amount'))                |    df.distinct(df.name, df.amount)                |
|                 |                                                                 |    df.distinct('name', 'amount')                  |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
|                 |                                                                 |                                                   |
| Reductions      |    df.amount.mean()                                             |    df.amount.mean()                               |
|                 |    df.amount.value_counts()                                     |    df.amount.count_values()                       |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
|                 | .. code-block:: python                                          | .. code-block:: python                            |
| Concatenate     |                                                                 |                                                   |
|                 |    pd.concat((df, df))                                          |    concat(df, df)                                 |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Column Type     | .. code-block:: python                                          | .. code-block:: python                            |
| Information     |                                                                 |                                                   |
|                 |    df.dtypes                                                    |    df.dshape                                      |
|                 |    df.amount.dtype                                              |    df.amount.dshape                               |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+

Blaze can simplify and make more readable some common IO tasks that one would want to do with pandas. These examples make use of the `odo <https://github.com/blaze/odo>`_ library. In many cases, blaze will able to handle datasets that can't fit into main memory, which is something that can't be easily done with pandas.


.. code-block:: python

   from odo import odo

+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Operation       | Pandas                                                          | Blaze                                             |
+=================+=================================================================+===================================================+
| Load            | .. code-block:: python                                          | .. code-block:: python                            |
| directory of    |                                                                 |                                                   |
| CSV files       |    df = pd.concat([pd.read_csv(filename)                        |    df = data('path/to/*.csv')                     |
|                 |                    for filename in                              |                                                   |
|                 |                    glob.glob('path/to/*.csv')])                 |                                                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Save result     | .. code-block:: python                                          | .. code-block:: python                            |
| to CSV file     |                                                                 |                                                   |
|                 |    df[df.amount < 0].to_csv('output.csv')                       |    odo(df[df.amount < 0],                         |
|                 |                                                                 |        'output.csv')                              |
|                 |                                                                 |                                                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
| Read from       | .. code-block:: python                                          | .. code-block:: python                            |
| SQL database    |                                                                 |                                                   |
|                 |    df = pd.read_sql('select * from t', con='sqlite:///db.db')   |    df = data('sqlite://db.db::t')                 |
|                 |                                                                 |                                                   |
|                 |    df = pd.read_sql('select * from t',                          |                                                   |
|                 |                     con=sa.create_engine('sqlite:///db.db'))    |                                                   |
+-----------------+-----------------------------------------------------------------+---------------------------------------------------+
