def pytest_addoption(parser):
    parser.addoption("--disable-tidl-offload", action="store_false", default=True)