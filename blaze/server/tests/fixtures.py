import pytest


def make_app_context_fixture(server):
    """
    Create an app_context fixture for a module.
    """

    @pytest.yield_fixture
    def app_context(request):
        """
        Fixture to mark that this test uses the flask application
        context.
        """
        with server.context() as ctx:
            yield ctx

    return app_context
