########################################################################
#
#       License: BSD
#       Created: February 1, 2011
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

"""
Default values for carray.

Feel free to change them for better adapting to your needs.

"""

import carray as ca


class Defaults(object):
    """Class to taylor the setters and getters of default values."""

    def __init__(self):
        self.choices = {}

        # Choices setup
        self.choices['eval_out_flavor'] = ("carray", "numpy")
        self.choices['eval_vm'] = ("numexpr", "python")

    def check_choices(self, name, value):
        if value not in self.choices[name]:
            raiseValue, "value must be either 'numexpr' or 'python'"

    #
    # Properties start here...
    #

    @property
    def eval_vm(self):
        return self.__eval_vm

    @eval_vm.setter
    def eval_vm(self, value):
        self.check_choices('eval_vm', value)
        if value == "numexpr" and not ca.numexpr_here:
            raise (ValueError,
                   "cannot use `numexpr` virtual machine "
                   "(minimum required version is probably not installed)")
        self.__eval_vm = value

    @property
    def eval_out_flavor(self):
        return self.__eval_out_flavor

    @eval_out_flavor.setter
    def eval_out_flavor(self, value):
        self.check_choices('eval_out_flavor', value)
        self.__eval_out_flavor = value


defaults = Defaults()


# Default values start here...

defaults.eval_out_flavor = "carray"
"""
The flavor for the output object in `eval()`.  It can be 'carray' or
'numpy'.  Default is 'carray'.

"""

defaults.eval_vm = "python"
"""
The virtual machine to be used in computations (via `eval`).  It can
be 'numexpr' or 'python'.  Default is 'numexpr', if installed.  If
not, then the default is 'python'.

"""

# If numexpr is available, use it as default
if ca.numexpr_here:
    defaults.eval_vm = "numexpr"

