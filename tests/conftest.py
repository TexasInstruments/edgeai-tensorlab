def pytest_addoption(parser):
    parser.addoption("--disable-tidl-offload", action="store_false", default=True)
    parser.addoption("--run-infer", action="store_true", default=False)