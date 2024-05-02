
# Redirect stdout to stderr to enable output capture for pytest-xdist
import sys
import pytest
from py.xml import html
import re
sys.stdout = sys.stderr


def pytest_addoption(parser):
    parser.addoption("--disable-tidl-offload", action="store_false", default=True)
    parser.addoption("--run-infer", action="store_true", default=False)
    parser.addoption("--no-subprocess", action="store_true", default=False)


# Adds the tidl_subgraphs attribute to test report
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if(report.when == "teardown"):
        regex_result = re.search("Offloaded Nodes - ([0-9]*)", report.capstdout)
        if(regex_result is None):
            report.tidl_subgraphs = "Not detected in test output"
        else:
            report.tidl_subgraphs = regex_result[1]


# Inserts the TIDL Subgraphs table header
def pytest_html_results_table_header(cells):
    cells.insert(2, html.th("TIDL Subgraphs"))

# Inserts the number of TIDL subgraphs for each row
def pytest_html_results_table_row(report, cells):
    if(hasattr(report,"tidl_subgraphs")):
        cells.insert(2, html.td(report.tidl_subgraphs))

