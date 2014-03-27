DataDescriptor as a 'Storage Descriptor' too
============================================

:Date: 2014-03-27

Rational
--------

The current situation of having all the storage properties centralized in the `Storage` class is getting the code unnecessarily complicated (see https://github.com/ContinuumIO/blaze/pull/205).  Some kind of specialization at the API level is highly desirable so that the specific flags for each storage can be dealt separately.

This proposal advocates for improving the different `DataDescriptor` classes for every supported format (currently DyND, BLZ, HDF5, CSV and JSON) so that they can deal with the storage capabilities internally.  This will actually render the `Storage` class useless and will effectively change the API of several important functions (array constructors and 'openers' mainly).

Also, besides of deprecating the `Storage` class, the different `DataDescriptors` will become first class citizens that the users will have to know about.

This document explains with detail the implications of the change.

The current API
---------------

The Blaze array constructors now look like::

  array(obj, dshape=None, caps={'efficient-write': True}, storage=None)
  empty(dshape, caps={'efficient-write': True}, storage=None)
  zeros(dshape, caps={'efficient-write': True}, storage=None)
  ones(dshape, caps={'efficient-write': True}, storage=None)

And the Blaze 'openers' of existing, persistent datasets::

  from_blz(persist, **kwargs)
  from_csv(persist, **kwargs)
  from_json(persist, **kwargs)
  from_hdf5(persist, datapath, **kwargs)

where **kwargs refers to the specific parameters depending on each format.

The new API
-----------

With this proposal, the array constructors will look like:

  array(obj, dshape=None, dd=None)
  empty(dshape, dd=None)
  zeros(dshape, dd=None)
  ones(dshape, dd=None)

And the openers::

  from_blz(dirpath, **kwargs)
  from_csv(filepath, **kwargs)
  from_json(filepath, **kwargs)
  from_hdf5(filepath, datapath, **kwargs)

where most of the `**kwargs` in the 'openers' will be passed to the `DataDescriptor` constructors (see below).

The DataDescriptor constructors (new in the public API)::

  DyNDDataDescriptor(dyndarr=None, new=True, **kwargs)
  BLZDataDescriptor(dirpath, new=True, **kwargs)
  CSVDataDescriptor(filepath, new=True, **kwargs)
  JSONDataDescriptor(filepath, new=True, **kwargs)
  HDF5DataDescriptor(filepath, datapath, new=True, **kwargs)

where `**kwargs` is where the user can set different parameters specific for the format (mode, appendable, compressed, CSV separator...).

Pros and cons of this proposal
------------------------------

Pros:

* There is a specialized DataDescriptor per each storage format. This provides a better way to deal with the specifics for each format.

* The `caps` parameter is not there anymore, so the constructors API is simplified.

Cons:

* The user will need to know about the kind of the format he will need, and Blaze will not decide for her anymore.

* That's a hard change in public API.

Other issues and considerations
-------------------------------

Right now, the `DataDescriptor` can only 'open' existing storage.  That means that we should add a new way to store the info for the data container before it could be created by the constructors (`array` and family) and fed by data.  

For this, a boolean `new` flag should be enough (if True, info for the new container should just be stored internally, if False, the persistent storage will be 'open'ed).  Then the constructors will need to check that the `new` attribute in DataDescriptor is set to True before trying to create the container.

In case the `dd` in constructors is set to 'None' then a `DyNDDataDescriptor` will be used.

Pending questions
-----------------

* The parameter name for the DataDescriptor instance in constructors should be `dd`, `data` or `datadescriptor`?  Probably the later.

* Provided that `DataDescriptors` will become first-class citizens, would not it be better to make the `Array._data` as public (i.e. `Array.data`)?  If so, should we change his name to match the one chosen in the question above?

* The `new` parameter in `DataDescriptor` is explicit enough?  And, it should have a 'True' value by default?  I think the answer for both questions should be 'yes'.
