'''
This script is used to run comparison between ONNXRT and TVMRT Compilers
Usage
    --runtime:          Specify the runtimes to run the tests for. If not specified, then run all the runtimes.
                        Accepted values are onnxrt, tvmrt

    --operator:         Specify the operators for test/compare. If not specified, then include all the operators.

    --compare:          Runs in compare mode. Doesn't run any test, uses the existing reports to generate
                        comparison report. Can specify the operators to compare.

Example usage:
    python3 run_operator_comparison.py --runtime tvmrt --operator Relu Max
        Runs the tvmrt runtine tests for Relu and Max
    
    python3 run_operator_comparison.py --operator Add
        Runs tests for Add for all the runtimes
    
    python3 run_operator_comparison.py --compare --operator Convolution
        Runs comparison for Convolution
'''

import csv
import subprocess
import os
import argparse
from bs4 import BeautifulSoup
import re

DEVICE = "AM68A"

## Change if changed in html
TIDL_OFFLOAD_INDEX = 3
TIDL_SUBGRAPH_INDEX = 2

# Add runtime if needed
ALL_RUNTIMES = ['onnxrt', 'tvmrt']
report = {
    "onnxrt": {},
    "tvmrt": {}
}

parser = argparse.ArgumentParser(description="Run operator tests and generate reports.")
parser.add_argument("--runtime", nargs="*", help="Runtime to test. Runs all if not specified")
parser.add_argument("--operator", nargs="*", help="Specify operators to test/compare. Runs all if not specified")
parser.add_argument("--compare", action="store_true", help="Runs in compare mode. Specify operators to compare and generate report")
args = parser.parse_args()

OPERATORS = []

# Load all operators
def load_operators(path):
    operators = []
    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            operators.append(dir_name)
        break
    return operators

if args.operator:
    OPERATORS = args.operator
else:
    OPERATORS = load_operators(os.path.abspath("../../tests/tidl_unit/tidl_unit_test_data/operator/"))

REPORT_DIR = "operator_test_report_comparison"
REPORT_PATH = os.path.abspath(REPORT_DIR)
OUT_DIR = "operator_test_report_comparison"

os.makedirs(OUT_DIR, exist_ok=True)

headers = [
    "Operator name",
    "Model Name",
    "ONNX Num Sub Graphs",      # Number of subgraphs detected/imported in ONNX
    "ONNXRT Functionality",     # ONNX Test Pass/Fail status
    "ONNX Error",               # Error if ONNX test failed
    "TVM Num Sub Graphs",       # Number of subgraphs detected/imported in TVM
    "TVM Functionality",        # TVM Test Pass/Fail status
    "TVM Error",                # Error if TVM test failed
    "Subgraphs Test",           # If number of subgraphs in ONNX and TVM matches
    "TVM Overall"               # Full functionality (test pass and number of subgraphs match)
]

rows = {
    "compile_with_nc": [],
    "compile_without_nc": [],
    "infer_with_nc": [],
    "infer_without_nc": [],
}

'''
Regex List, used to extract errors captured.
Regex can either of these
    A regex str r"<regex>", which will extract the whole line
    A dictionary
        {
            "regex" : <regex to match>,
            "error" : <error message>
        }

The checks go from top to bottom and will terminate once an error is found.
'''
error_regex = [
    r"\nInternalError: (.*)",
    r"\nTVMError: (.*)",
    r"\n(.*) \[ONNXRuntimeError\] (.*)",
    r"\nAssertionError: (.*)",
    r"\nFailed:  (.*)",
    { "regex": r"DLRError", "error": "Compilation Failed" },
    { "regex": r"stopped exitcode=-SIGBUS>", "error": "Bus Error"}
]

def extract_report_data(runtime, operator, report_dict):
    '''
    Extracts data from html report and store it in report_dict.
    report_dict structure:
    {
        "compile_with_nc": [
            {
                "node_id": <pytest_node_id>,
                "name": <model name>,
                "tidl_subgraphs": <number of subgraphs in report>,
                "tidl_offload": <whether offloaded to tidl>,
                "outcome": <test pass, fail or xfail>,
                "error" <error captured through regex>
            },
            ...
        ],
        "compile_without_nc": [...],
        "infer_with_nc": [...],
        "infer_without_nc": [...],
    }
    Prints summary (total, pass, fail) after each operator reports are extracted
    '''
    # Reports are store in $REPORT_PATH/$runtime/$operator
    folder_path = os.path.join(REPORT_PATH, runtime, operator)
    #If operator test hasn't been run
    if not os.path.exists(folder_path):
        return

    passing, total = 0, 0
    for file in os.listdir(folder_path):
        if file.endswith(".html"):
            with open(os.path.join(folder_path, file), 'r') as f:
                # data = json.load(f)
                soup = BeautifulSoup(f, "html.parser")
                table = soup.find("table", {"id": "results-table"})

                rows = table.find_all("tbody")

                name = file.split('.')[0]
                if name not in report_dict:
                    report_dict[name] = []
                
                for row in rows:
                    cells = row.find_all("td")
                    total += 1
                    if cells[0].text.strip() == "Passed":
                        passing += 1
                    error = "-"
                    err_str = cells[6].text.strip()

                    for error_reg in error_regex:
                        if error != "-":
                            break
                        if isinstance(error_reg, dict):
                            error = error_reg["error"] if re.search(error_reg["regex"], err_str) is not None else "-"
                        else:
                            error_match = re.search(error_reg, err_str) 
                            if error_match is not None:
                                if error_reg.startswith(r"\n"):
                                    error = error_match[0][1:]
                                else:
                                    error = error_match[0]
                    
                    report_dict[name].append({
                        "nodeid": cells[1].text.strip(),
                        "name": (cells[1].text.strip()).split('[')[-1][:-1],
                        "tidl_subgraphs": cells[TIDL_SUBGRAPH_INDEX].text.strip(),
                        "tidl_offloads": cells[TIDL_OFFLOAD_INDEX].text.strip(),
                        "outcome": cells[0].text.strip(),
                        "error": error
                    })
    print(f"│  Runtime: \x1b[36m{runtime}\x1b[0m")
    print(f"│   \x1b[93m Total tests: {total}\x1b[0m,       \x1b[92m Passed tests: {passing}\x1b[0m,        \x1b[91m Failed tests: {total - passing}\x1b[0m")

# Runs tests if not in compare mode
if not args.compare:
    print("──────────────────────────── Running Testing Script ────────────────────────────")
    script = [
        'bash',
        'run_operator_test.sh',
        DEVICE,
    ]
    if args.operator and len(args.operator):
        script += ['--operators'] + args.operator

    if args.runtime and len(args.runtime):
        script += ['--runtimes'] + args.runtime
    else:
        script += ['--runtimes'] + ALL_RUNTIMES
    
    subprocess.run(script, text=True)

# Extract data from reports
print("──────────────────────────── Extracting data from reports ────────────────────────────")
for operator in OPERATORS:
    print(f"┌────────────────────────── Results for \x1b[95m{operator}\x1b[0m")
    for runtime in ALL_RUNTIMES:
        extract_report_data(runtime, operator, report[runtime])
    print("└─────────────────────────────────────────────────────────────────────────────")

# Format and prepare data for csv
for op, onnx_res in report["onnxrt"].items():
    tvm_res = report["tvmrt"].get(op, None)
    if not tvm_res:
        continue
    for res in onnx_res:
        t = list(filter(lambda x: x['nodeid'] == res['nodeid'], tvm_res))
        if not len(t):
            continue
        t = t[0]
        data = [
            res['name'].split('_')[0],
            res['name'],
            res['tidl_subgraphs'],
            res['outcome'],
            res['error'],
            t['tidl_subgraphs'],
            t['outcome'],
            t['error']
        ]
        data.append('Passing' if data[2] == data[5] else 'Failing')
        data.append('Passing' if (data[8] == 'Passing' and data[6] == 'Passed') else 'Failing')
        rows[op].append(data)

# Individual comparison report (either inference or compilation)
print("Writing CSV Comparisions >")
for op, report in rows.items():
    with open(os.path.join(OUT_DIR, f"{op}_comparison.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows[op])

complete = {
    "with_nc": [],
    "without_nc": []
}

complete_headers = [
    "Operator Name",
    "Model Name",
    "ONNX Num Sub Graphs",              # Number of subgraphs detected/imported in ONNX
    "ONNXRT Compilation Status",        # ONNX Compilation Test Pass/Fail status
    "ONNX Inference Status",            # ONNX Inference Test Pass/Fail status
    "TVM Subgraphs",                    # Number of subgraphs detected/imported in ONNX
    "TVM Compilation Status",           # TVM Compilation Test Pass/Fail status
    "TVM Subgraph Status",              # If number of subgraphs in ONNX and TVM matches
    "TVM Compilation Overall",          # TVM Compilation Overall functionality (TVM Compilation Test pass and Subgraphs Test Pass)
    "TVM Inference Status",             # TVM Inference Test Pass/Fail status
    "Compilation TVM Error",            # TVM Compilation Error if test failed
    "Inference TVM Error",              # TVM Inference Error if test failed
    "Overall"                           # Overall functionality (Compilation Overall and Inference Pass)
]

# Complete comparison report
print("Writing Complete Comparisions >")
for k in complete.keys():
    # Get compilation and inference results of both compilation and inference of with/without nc
    inference = rows[f"infer_{k}"]
    compilation = rows[f"compile_{k}"]
    complete_rows = []
    for comp in compilation:
        infer = list(filter(lambda x: x[1] == comp[1], inference))
        if not len(infer):
            continue
        infer = infer[0]
        row = [
            comp[0],
            comp[1],
            comp[2],
            comp[3],
            infer[3],
            comp[5],
            comp[6],
            comp[8],
            comp[9],
            infer[6],
            comp[7],
            infer[7]
        ]
        overall = "-"
        if row[8] == "Passing" and row[9] == "Passed":
            overall = "Passing"
        elif row[8] != "Passing":
            overall = "Compilation Failure"
        elif row[9] != "Passed":
            overall = "Inference Failure"
        row.append(overall)
        complete_rows.append(row)

    with open(os.path.join(OUT_DIR, f"{k}_comparison.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(complete_headers)
        writer.writerows(complete_rows)