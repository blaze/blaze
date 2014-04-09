
from contextlib import contextmanager
import tempfile
import os

@contextmanager
def filetext(text, extension='.csv'):
    # write text to hidden file
    handle, filename = tempfile.mkstemp(extension)
    with os.fdopen(handle, "w") as f:
        f.write(text)

    # Yield control to test
    yield filename

    # Clean up the written file
    os.remove(filename)

