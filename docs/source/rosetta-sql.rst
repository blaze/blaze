SQL to Blaze
============

This page maps SQL expressions to blaze expressions.

.. note::

   The following SQL expressions are somewhat specific to PostgreSQL, but blaze
   itself works with any database for which a SQLAlchemy dialect exists.

Prerequisites
-------------

If you're interested in testing these against a PostgreSQL database, make sure
you've executed the following code in ``psql`` session:

.. code-block:: sql

   CREATE TABLE df (
       id BIGINT,
       amount DOUBLE PRECISION,
       name TEXT
   );

On the blaze side of things, the table below assumes the following code has
been executed:

.. code-block:: python

   >>> from blaze import symbol, by, join, concat
   >>> df = symbol('df', 'var * {id: int64, amount: float64, name: string}')

.. note::

   Certain SQL constructs such as window functions don't directly correspond to
   a particular Blaze expression. ``Map`` expressions are the closest
   representation of window functions in Blaze.


+-----------------+-------------------------------------------------+-----------------------------------------+
| Computation     | SQL                                             | Blaze                                   |
+=================+=================================================+=========================================+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
| Column          |                                                 |                                         |
| Arithmetic      |    select amount * 2 from df                    |    df.amount * 2                        |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
| Multiple        |                                                 |                                         |
| Columns         |    select id, amount from df                    |    df[['id', 'amount']]                 |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
| Selection       |    selelct * from df where amount > 300         |    df[df.amount > 300]                  |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|  Group By       |                                                 |                                         |
|                 |    select avg(amount) from df group by name     |    by(df.name, amount=df.amount.mean()) |
|                 |                                                 |                                         |
|                 +-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
|                 |    select avg(amount) from df group by name, id |    by(merge(df.name, df.id),            |
|                 |                                                 |       amount=df.amount.mean())          |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
| Join            |                                                 |                                         |
|                 |    select * from                                |    join(df, df2, 'name')                |
|                 |        df inner join df2                        |                                         |
|                 |    on df.name = df2.name                        |                                         |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
| Map             |    select amount + 1 over () from df            |    df.amount.map(lambda x: x + 1,       |
|                 |                                                 |                  'int64')               |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
| Relabel Columns |    select                                       |    df.relabel(name='alias',             |
|                 |        id,                                      |               amount='dollars')         |
|                 |        name as alias,                           |                                         |
|                 |        amount as dollars                        |                                         |
|                 |     from df                                     |                                         |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
| Drop duplicates |    select distinct * from df                    |    df.distinct()                        |
|                 +-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
|                 |    select distinct(name) from df                |    df.name.distinct()                   |
|                 +-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block::python                   |
|                 |                                                 |                                         |
|                 |    /* postgresql only */                        |    # postgresql only                    |
|                 |    select distinct on (name) * from             |    df.sort(df.name).distinct(df.name)   |
|                 |    df order by name                             |    df.sort('name').distinct('name')     |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
| Reductions      |    select avg(amount) from df                   |    df.amount.mean()                     |
|                 +-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
|                 |    select amount, count(amount)                 |    df.amount.count_values()             |
|                 |    from df group by amount                      |                                         |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
| Concatenate     |    select * from df                             |    concat(df, df)                       |
|                 |    union all                                    |                                         |
|                 |    select * from df                             |                                         |
+-----------------+-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
|                 |    select                                       |    df.dshape                            |
|                 |        column_name,                             |                                         |
|                 |        data_type,                               |                                         |
|                 |        character_maximum_length                 |                                         |
|                 |    from                                         |                                         |
|                 |        information_schema.columns               |                                         |
| Column Type     |    where                                        |                                         |
| Information     |        table_name = 'df'                        |                                         |
|                 +-------------------------------------------------+-----------------------------------------+
|                 | .. code-block:: sql                             | .. code-block:: python                  |
|                 |                                                 |                                         |
|                 |    select                                       |    df.amount.dshape                     |
|                 |        column_name,                             |                                         |
|                 |        data_type,                               |                                         |
|                 |        character_maximum_length                 |                                         |
|                 |    from                                         |                                         |
|                 |        information_schema.columns               |                                         |
|                 |    where                                        |                                         |
|                 |        table_name = 'df'                        |                                         |
|                 |            and                                  |                                         |
|                 |        column_name = 'amount'                   |                                         |
+-----------------+-------------------------------------------------+-----------------------------------------+
