import pytest


def make_app_context_fixture(server):
    """
    Create an app_context fixture for a module.
    """

    @pytest.fixture(scope='function')
    def app_context(request):
        """
        Fixture to mark that this test uses the flask application
        context.
        """
        ctx = server.context()
        request.addfinalizer(lambda: ctx.__exit__(None, None, None))
        return ctx.__enter__()

    return app_context
