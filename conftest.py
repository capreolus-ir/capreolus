def pytest_addoption(parser):
    parser.addoption("--check-download", action="store_true", dest="download", default=False, help="enable tests marked download")


def pytest_configure(config):
    if not config.option.download:
        markexpr = "not download"
        if hasattr(config.option, "markexpr") and getattr(config.option, "markexpr"):
            markexpr = getattr(config.option, "markexpr") + " and " + markexpr
        setattr(config.option, "markexpr", markexpr)
