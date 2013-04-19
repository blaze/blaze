from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

from llvm.core import Function

def dump_cfg(function):
    assert isinstance(function, Function)
    llvm_ir = str(function)

    with NamedTemporaryFile(delete=True) as tempbc:
        with NamedTemporaryFile(delete=True) as tempdot:
            tempbc.write(llvm_ir)
            tempdot.flush()

            p = Popen(['llvm-as -f', PIPE])
            p.wait()

            p = Popen(['dot','-Tpng',tempdot.name,'-o','%s.png' % fname])
            p.wait()

            assert p.returncode == 0

            # Linux
            p = Popen(['feh', '%s.png' % fname])
            # Macintosh
            #p = Popen(['open', fname])
