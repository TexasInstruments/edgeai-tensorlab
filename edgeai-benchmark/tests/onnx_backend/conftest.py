
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
    report.tidl_subgraphs = "Not detected"
    if report.when == 'call' or report.when == 'teardown':
        regex1_result = re.search("Final number of subgraphs created are : ([0-9]*)", report.capstdout)
        if(regex1_result is None):
            regex2_result = re.search(r"\|\s*C7x\s*\|\s*\d+\s*\|\s*(\d+|x)\s*\|", report.capstdout)
            if (regex2_result is not None):
                report.tidl_subgraphs = regex2_result[1]
        else:
            report.tidl_subgraphs = regex1_result[1]


# Inserts the TIDL Subgraphs table header
def pytest_html_results_table_header(cells):
    cells.insert(2, html.th("TIDL Subgraphs"))

# Inserts the number of TIDL subgraphs for each row
def pytest_html_results_table_row(report, cells):
    if(hasattr(report,'tidl_subgraphs')):
        cells.insert(2, html.td(report.tidl_subgraphs))