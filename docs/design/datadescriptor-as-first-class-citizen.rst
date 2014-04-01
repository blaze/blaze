DataDescriptor as a 'Storage Descriptor' too
============================================


Rational
--------

The current situation of having all the storage properties centralized
in the `Storage` class is getting the code unnecessarily complicated
(see https://github.com/ContinuumIO/blaze/pull/205).  Some kind of
specialization at the API level is highly desirable so that the
specific flags for each storage can be dealt separately.

This proposal advocates for improving the different `DataDescriptor`
classes for every supported format (currently DyND, BLZ, HDF5, CSV and
JSON) so that they can deal with the storage capabilities internally.
This will actually render the `Storage` class useless and will
effectively change the API of several important functions (array
constructors and 'openers' mainly).

Also, besides of deprecating the `Storage` class, the different
`DataDescriptors` (renamed to `DDesc`, for brevity) will become first
class citizens that the users will have to know about.

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

  array(obj, dshape=None, ddesc=None)
  empty(dshape, ddesc=None)
  zeros(dshape, ddesc=None)
  ones(dshape, ddesc=None)

And for opening an existing dataset::

  ddesc = BLZ_DDesc(path, **kwargs) # or
  ddesc = CSV_DDesc(path, **kwargs) # or
  ddesc = JSON_DDesc(path, **kwargs) # or
  ddesc = HDF5_DDesc(path, datapath, **kwargs) # followed by
  array(ddesc=ddesc)  # or just
  array(ddesc)

where most of the `**kwargs` in the data descriptors will be passed to
the underlying libraries in charge of handling the format details
(BLZ, HDF5, CSV, JSON..).

Also, as the different `DDesc` interfaces will be public, exposing it
from the `Array` object will be possible::

  Array.ddesc -> the DDesc associated with the Blaze `Array`

Pros and cons of this proposal
------------------------------

Pros:

* There is a specialized DDesc per each storage format. This provides
  a better way to deal with the specifics for each format.

* The `caps` parameter is not there anymore, so the constructors API
  is simplified.

Cons:

* The user will need to know about the kind of the format he will
  need, and Blaze will not decide for her anymore.

* That's a hard change in public API.

Other issues and considerations
-------------------------------

Right now, the `DDesc` can only 'open' existing storage. That means
that we should add a new way to store the info for the data container
before it could be created by the constructors (`array` and family)
and fed by data [DONE].

Also, in case the `ddesc` parameter in constructors is set to 'None'
then a `DyNDDataDescriptor` will be used [DONE].
