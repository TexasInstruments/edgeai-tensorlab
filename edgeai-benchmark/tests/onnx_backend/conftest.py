
# Redirect stdout to stderr to enable output capture for pytest-xdist
import sys
import pytest
from py.xml import html
import re
import os
from datetime import datetime

def pytest_addoption(parser):
    parser.addoption("--disable-tidl-offload", action="store_true")
    parser.addoption("--run-infer", action="store_true", default=False)
    parser.addoption("--no-subprocess", action="store_true", default=False)

# Configures html report name and path
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.option.htmlpath = 'logs/report_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")+".html"

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

