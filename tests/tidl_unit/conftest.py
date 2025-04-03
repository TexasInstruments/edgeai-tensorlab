
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

def pytest_sessionfinish(session):
    plugin = session.config._json_report
    json_path = session.config.option.htmlpath.replace(".html",".json")
    plugin.save_report(json_path)

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
    report.complete_tidl_offload = "Not detected"
    if report.when == 'call' or report.when == 'teardown':
        # Parsing subgraphs
        num_subgraph_regex = re.search("Final number of subgraphs created are : ([0-9]*)", report.capstdout)
        if(num_subgraph_regex is None):
            c7x_table_regex = re.search(r"\|\s*C7x\s*\|\s*\d+\s*\|\s*(\d+|x)\s*\|", report.capstdout)
            if (c7x_table_regex is not None):
                report.tidl_subgraphs = c7x_table_regex[1]
        else:
            report.tidl_subgraphs = num_subgraph_regex[1]

        if report.tidl_subgraphs.isdigit() and int(report.tidl_subgraphs) >= 1:
            # Parsing complete tidl offload
            total_nodes_regex = re.search("Total Nodes - ([0-9]*)", report.capstdout)
            offloaded_nodes_regex = re.search("Offloaded Nodes - ([0-9]*)", report.capstdout)
            if(total_nodes_regex is None or offloaded_nodes_regex is None):
                cpu_table_regex = re.search(r"\|\s*CPU\s*\|\s+(\d+)\s+\|", report.capstdout)
                if (cpu_table_regex is not None):
                    if str(cpu_table_regex[1]).lower() == '0':
                        report.complete_tidl_offload = "True"
                    else:
                        report.complete_tidl_offload = "False"
            elif (total_nodes_regex is not None and offloaded_nodes_regex is not None):
                try:
                    total_nodes = int(total_nodes_regex[1].strip())
                    offloaded_nodes = int(offloaded_nodes_regex[1].strip())
                    if (offloaded_nodes >= total_nodes):
                        report.complete_tidl_offload = "True"
                    else:
                        report.complete_tidl_offload = "False"
                except:
                    pass
        else:
            report.complete_tidl_offload = "-"
# Inserts the TIDL Subgraphs table header
def pytest_html_results_table_header(cells):
    cells.insert(2, html.th("TIDL Subgraphs"))
    cells.insert(3, html.th("Complete TIDL Offload"))

# Inserts the number of TIDL subgraphs for each row
def pytest_html_results_table_row(report, cells):
    if(hasattr(report,'tidl_subgraphs')):
        cells.insert(2, html.td(report.tidl_subgraphs))
    if(hasattr(report,'complete_tidl_offload')):
        cells.insert(3, html.td(report.complete_tidl_offload))