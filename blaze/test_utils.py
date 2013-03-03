import sys
import os
import shutil
import tempfile

class assert_raises(object):

    def __init__(self, exc_ty):
        self.exc_ty = exc_ty

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        if type is None:
            raise AssertionError('Exception not raised')
        if issubclass(type, self.exc_ty):
            return True
        raise AssertionError('Wrong exception raised: ' + repr(value))

class temp_dir(object):
    '''Context manager that creates a temporary directory and automatically
    removes the directory and its contents at the end of the block.

    To assist in debugging failing tests, if the environment variable
    BLAZE_TEST_KEEP_TEMP is set to 1, then the context manager will not delete
    the temporary directory.  Instead, the name of the directory will be
    printed to stderr.

    Example usage:

        with temp_dir() as tmpd:
            temp_filename = os.path.join(tmpd, 'test.txt')
            with open(temp_filename, 'w') as f:
                f.write('Test Output')
    '''

    def __init__(self):
        pass

    def __enter__(self):
        self.dirname = tempfile.mkdtemp(prefix='blaze_test_')
        return self.dirname

    def __exit__(self, type, value, traceback):
        if not bool(os.environ.get('BLAZE_TEST_KEEP_TEMP', False)):
            shutil.rmtree(self.dirname)
        else:
            print >>sys.stderr, '\nKeeping temporary directory:', self.dirname

        return False  # If there was an exception, reraise it.
