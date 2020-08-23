def pytest_addoption(parser):
    parser.addoption("--check-download", action="store_true", dest="download", default=False, help="enable tests marked download")


def pytest_configure(config):
    config.addinivalue_line("markers", "download: slow tests that test downloading a dataset")

    if not config.option.download and "download" not in config.option.markexpr.split():
        markexpr = "not download"
        if config.option.markexpr:
            markexpr = config.option.markexpr + " and " + markexpr
        config.option.markexpr = markexpr
