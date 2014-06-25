Catalog
=======

.. highlight:: python

The Blaze catalog is a way to organize and work with your data. In
the current version, it is implemented for a local catalog on one machine,
and in future revisions it will be extended to be consistent across
Blaze clusters.

Guided Tour
-----------

By default, when you start Blaze for the first time, an empty catalog
is created for you as a place to put your data. The default catalog
is configured at ``~/.blaze/catalog.config``, and it is set to point
to ``~/Arrays`` as the source of array data. This is not created
automatically by Blaze, but is easy to initialize by running.::

    In [1]: blaze.catalog.init_default()


Loading Blaze in IPython
~~~~~~~~~~~~~~~~~~~~~~~~

The Blaze source repository includes a small set of sample arrays
in a catalog, we will load and work with these arrays to illustrate
the features. To start, load an IPython prompt, and import Blaze.::

    In [1]: import blaze

The first thing to do, since we want to demonstrate our sample catalog,
is to load that catalog's configuration.::

    In [2]: blaze.catalog.load_config(
                r"~/blaze/samples/server/sample_arrays.yaml")

To see information about the catalog that is currently loaded,
simply print out the repr of the catalog configuration object.::

    In [3]: blaze.catalog.config
    Out[3]: 
    Blaze Catalog Configuration
    /path/to/sample_arrays.yaml

    root: /path/to/Arrays

Navigating the Catalog
~~~~~~~~~~~~~~~~~~~~~~

On import, because we are in an IPython environment, Blaze has
configured some line magics to work with the catalog. These are
similar to the usual unix ``ls``, ``cd``, and ``pwd``, but prefixed
with ``b`` for Blaze. Let's start by taking a look what our working
directory is in the Blaze catalog, and what arrays and directories
are available. Note that you may have to say ``%bpwd`` instead of
``bpwd`` in some circumstances, but we will avoid the percent and
rely on IPython's magic.::

    In [5]: bpwd
    Out[5]: '/'

    In [6]: bls
    btc dates_vals kiva kiva_tiny

In the latter example, the directories should appear in a different
color than the arrays. You may see that ``btc``, ``kiva``, and
``kiva_tiny`` are directories, while ``dates_vals`` is an array.
Let's switch to the ``kiva_tiny`` directory.::

    In [8]: bcd kiva_tiny
    /kiva_tiny

    In [9]: bpwd
    Out[9]: u'/kiva_tiny'

    In [10]: bls
    lenders loans

Loading Array Data
~~~~~~~~~~~~~~~~~~

There are two arrays available here. Let's load the ``lenders`` and
take a look at its data. This is done with the ``%barr`` magic,
which can be used on the right side of an assignment statement.::

    In [11]: x = %barr lenders

    In [12]: x.dshape
    Out[12]: dshape("52, { lender_id : json; name : json; image : { id : int64; template_id : int64 }; whereabouts : string; country_code : json; uid : string; member_since : string; personal_url : string; occupation : string; loan_because : string; occupational_info : string; loan_count : int64; invitee_count : int64; inviter_id : json }")

    In [15]: x.occupation
    Out[15]: 
    array(
    [u'International Development Manager', u'retired dentist', u'', u'', u'',
     u'internet campaigner', u'Engineer', u'Nursery Manager', u'Web designer',
     u'Technical Assistant', u'', u'', u'', u'', u'Licenses Optician',
     u'physician', u'', u'Lic. Optician', u'', u'', u'', u'Master',
     u'Politician', u'guardian', u'', u'', u'Carpenter', u'', u'IT',
     u'Art Director', u'Nomad', u'Student', u'Teacher',
     u'IT Network consultant', u'Pastry Chef', u'Student', u'',
     u'Writer and Editor', u'Union Construction', u'veterinarian',
     u'Architect', u'Technical Trainer', u'pharmacist / farmaceutico',
     u'Marketing Manager', u'teacher', u'', u'', u'', u'', u'retired',
     u'Sales Director', u''],
          dshape='52, string')

The Array Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~

Let's now take a look at what the files on disk look like, backing
the array we loaded. The ``lenders`` array is described by a file ``lenders.array``,
as follows.::

    In [17]: print open(r"~/blaze/samples/server/arrays/kiva_tiny/lenders.array").read()
    type: json
    import: {}
    datashape: |
        var, {
            lender_id: json; # Option(string);
            name: json; # Option(string);
            image: {
                id: int64;
                template_id: int64;
            };
            whereabouts: string;
            country_code: json; # string(2);
            uid: string;
            member_since: string; # datetime<minutes>
            personal_url: string; # URL type?
            occupation: string;
            loan_because: string;
            occupational_info: string;
            loan_count: int64;
            invitee_count: int64;
            inviter_id: json; # sometimes string, sometimes number 0. 0 is being used as the "missing value"
        }

This file is a YAML configuration file, which describes some data import
settings including the datashape of the data. In this case, the data is
in JSON format (``lenders.json`` right beside the ``.array`` file), and
its datashape is provided using YAML's multi-line string literal syntax.

If we go back up one level to the root of the catalog, we can see another
array ``dates_vals``, which is in CSV format. Here is how it looks.::

    In [18]: bcd ..
    /

    In [19]: x = %barr dates_vals

    In [19]: x[:10]
    Out[19]: 
    array(
    [{u'values': 0, u'data': datetime.date(2013, 1, 1)},
     {u'values': 1, u'data': datetime.date(2013, 1, 2)},
     {u'values': 2, u'data': datetime.date(2013, 1, 3)},
     {u'values': 3, u'data': datetime.date(2013, 1, 4)},
     {u'values': 4, u'data': datetime.date(2013, 1, 5)},
     {u'values': 5, u'data': datetime.date(2013, 1, 6)},
     {u'values': 6, u'data': datetime.date(2013, 1, 7)},
     {u'values': 7, u'data': datetime.date(2013, 1, 8)},
     {u'values': 8, u'data': datetime.date(2013, 1, 9)},
     {u'values': 9, u'data': datetime.date(2013, 1, 10)}],
          dshape='10, { data : date; values : int32 }')

