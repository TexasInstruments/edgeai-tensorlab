'''
This script is used to run comparison between ONNXRT and TVMRT Compilers
Usage
    --runtime:          Specify the runtimes to run the tests for. If not specified, then run all the runtimes.
                        Accepted values are onnxrt, tvmrt

    --operator:         Specify the operators for test/compare. If not specified, then include all the operators.

    --compare:          Runs in compare mode. Doesn't run any test, uses the existing reports to generate
                        comparison report. Can specify the operators to compare.

    --summary:          Used in compare mode. Generates summary_report.csv for each runtime.

    --include_sg_fails  Modify tvmrt report summary to include offload failures for each operator.

Example usage:
    python3 run_operator_comparison.py --runtime tvmrt --operator Relu Max
        Runs the tvmrt runtine tests for Relu and Max
    
    python3 run_operator_comparison.py --operator Add
        Runs tests for Add for all the runtimes
    
    python3 run_operator_comparison.py --compare --operator Conv
        Runs comparison for Conv

If single tests are to be run, then only that should be passed. Example:
    python3 run_operator_comparison.py --operator MaxPool_2
        Runs onnx and tvm tests for just MaxPool_2 test
'''

import csv
import subprocess
import os
import argparse
from bs4 import BeautifulSoup
import re

DEVICE = "AM68A"
TIDL_EXTRACTED_TOOLS_PATH = "<path to extracted tidl tools folder>"
TIDL_TOOLS_PATH = "<path to tidl tools tarball file"

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
parser.add_argument("--summary", action="store_true", help="Generates summary from the existing reports")
parser.add_argument("--include_sg_fails", action="store_true", help="Modify tvmrt summary reports to include Offload failures")
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
    OPERATORS = load_operators(os.path.abspath("../tidl_unit_test_data/operators/"))

REPORT_DIR = "operator_test_reports"
REPORT_PATH = os.path.abspath(REPORT_DIR)
OUT_DIR = "operator_test_reports/comparison"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

# Headers for individual comparison report
headers = [
    "Operator name",
    "Model Name",
    "ONNX Subgraphs",             # Number of subgraphs detected/imported in ONNX
    "ONNX Complete Offload",      # Complete offload to ONNX (if all nodes are offloaded)
    "ONNXRT Functionality",       # ONNX Test Pass/Fail status
    "ONNX Error",                 # Error if ONNX test failed
    "TVM Subgraphs",              # Number of subgraphs detected/imported in TVM
    "TVM Complete Offload",       # Complete offload to TVM (if all nodes are offloaded)
    "TVM Functionality",          # TVM Test Pass/Fail status
    "TVM Error",                  # Error if TVM test failed
    "Offload Test",               # If complete offload matches in TVM and ONNX
    "TVM Overall",                # Full functionality (test pass and number of subgraphs match)
    "Comments"                    # Comments for a test case, if any
]

# Headers for complete comparison report
complete_headers = [
    "Operator Name",
    "Model Name",
    "ONNX Subgraphs",                   # Number of subgraphs detected/imported in ONNX
    "ONNX Complete Offload",            # Complete offload to ONNX (if all nodes are offloaded)
    "ONNXRT Compilation Status",        # ONNX Compilation Test Pass/Fail status
    "ONNX Inference Status",            # ONNX Inference Test Pass/Fail status
    "TVM Subgraphs",                    # Number of subgraphs detected/imported in ONNX
    "TVM Complete Offload",             # Complete offload to TVM (if all nodes are offloaded)
    "TVM Compilation Status",           # TVM Compilation Test Pass/Fail status
    "TVM Offload Test",                 # If complete offload in ONNX and TVM matches
    "TVM Compilation Overall",          # TVM Compilation Overall functionality (TVM Compilation Test pass and Offload Test Pass)
    "TVM Inference Status",             # TVM Inference Test Pass/Fail status
    "Compilation TVM Error",            # TVM Compilation Error if test failed
    "Inference TVM Error",              # TVM Inference Error if test failed
    "Overall",                          # Overall functionality (Compilation Overall and Inference Pass)
    "Comments"                          # Comments for a test case, if any
]

rows = {
    "compile_with_nc": [],
    "compile_without_nc": [],
    "infer_ref_with_nc": [],
    "infer_ref_without_nc": [],
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
    { "regex": r"\nInvoke  : ERROR: Unable to open network file", "error": "Compilation Failed / Network File not found"},
    r"\nInternalError: (.*)",
    r"\nTVMError: (.*)",
    r"\n(.*) \[ONNXRuntimeError\] (.*)",
    r"\nAssertionError: (.*)",
    r"\nFailed:  (.*)",
    r"\nValueError: (.*)",
    r"\n\[TVM Optimize\]: (.*)",
    r"\ntvm.error.OpAttributeInvalid: (.*)",
    { "regex": r"DLRError", "error": "Compilation Failed" },
    { "regex": r"stopped exitcode=-SIGBUS>", "error": "Bus Error"},
    { "regex": r"stopped exitcode=-SIGSEGV>", "error": "Segmentation Fault"},
    { "regex": r"assert None == 0", "error": "Timeout"}
]


'''
Ignore errors for specific operators, and modify test cases to be passed in CSV
Format:
{
    "ONNX Operator Name": [
        {
            "regex": r"<regex to match>",
            "ignore": [<list of keys to modify to 'Passed'>]
            "message"?: "<message to print in comments>",
        },
        ...
    ]
}
"message" is optional, if not provided, the regex match will be used as comment.
"ignore" is a list of keys in the report to modify to "Passed" if the regex matches.
Keys that can be modified are:
    - "TVM Offload Test"
    - "TVM Compilation Status"
    - "TVM Inference Status"
'''
ignores_dict = {
    "Resize": [
        {
            "regex": r"\[TVM Optimize\]: (.*)",
            "ignore": ["TVM Offload Test"],
            "message": "Removed Identity Resize node from graph"
        },
        {
            "regex": r"tvm.error.OpAttributeInvalid: \[Unsupported\] (.*)",
            "ignore": ["TVM Offload Test", "TVM Compilation Status", "TVM Inference Status"]
        },
        {
            "regex": r"ValueError: Unsupported (.*)",
            "ignore": ["TVM Offload Test", "TVM Compilation Status", "TVM Inference Status"]
        }
    ]
}

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
        "infer_ref_with_nc": [...],
        "infer_ref_without_nc": [...],
    }
    Prints summary (total, pass, fail) after each operator reports are extracted
    '''
    # Reports are stored in $REPORT_PATH/$runtime/$operator
    folder_path = os.path.join(REPORT_PATH, runtime, DEVICE, operator)
    #If operator test hasn't been run
    if not os.path.exists(folder_path):
        return

    passing, total, ignored = 0, 0, 0
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
                    
                    data = {
                        "nodeid": cells[1].text.strip(),
                        "name": (cells[1].text.strip()).split('[')[-1][:-1],
                        "tidl_subgraphs": cells[TIDL_SUBGRAPH_INDEX].text.strip(),
                        "tidl_complete_offload": cells[TIDL_OFFLOAD_INDEX].text.strip() if cells[TIDL_OFFLOAD_INDEX].text.strip() != "-" else "False",
                        "outcome": cells[0].text.strip(),
                        "error": error
                    }

                    # Check if it can be ignored
                    op = data["name"].split("_")[0]
                    if op in ignores_dict:
                        for ignore in ignores_dict[op]:
                            if ("infer" in name and "TVM Inference Status" in ignore["ignore"]) or ("compile" in name and "TVM Compilation Status" in ignore["ignore"]):
                                err = re.search(ignore["regex"], err_str)
                                if err is not None:
                                    ignored += 1
                                    data["outcome"] = "Passed"
                                    if "message" in ignore:
                                        data["comment"] = "[IGNORED] " + ignore["message"]
                                    else:
                                        data["comment"] = "[IGNORED] " + str(err[0])
                                    break
                        
                    report_dict[name].append(data)
    print(f"│  Runtime: \x1b[36m{runtime}\x1b[0m")
    print(f"│   \x1b[93m Total tests: {total}\x1b[0m,       \x1b[92m Passed tests: {passing + ignored}\x1b[0m,        \x1b[91m Failed tests: {total - passing - ignored}", f"\x1b[38:5:209m(Ignored: {ignored})\x1b[0m" if ignored else "\x1b[0m")

# Runs tests if not in compare mode
if not args.compare:
    print("──────────────────────────── Running Testing Script ────────────────────────────")

    # If a single test is run, run it directly from here
    if (len(OPERATORS) == 1) and re.search(r"^(.*)_(\d*)$", OPERATORS[0]):
        import shlex
        import shutil

        LOGS_PATH = "../logs"
        envs=["ARM64_GCC_PATH", "TIDL_RT_ONNX_VARDIM", "TIDL_RT_DDR_STATS", "TIDL_RT_DDR_STATS", "TIDL_RT_PERFSTATS", "TIDL_RT_AVX_REF", "TIDL_ARTIFACT_SYMLINKS"]

        command = shlex.split("bash -c 'cd ../../../ && source ./run_set_env.sh && env'")
        proc = subprocess.Popen(command, stdout = subprocess.PIPE, universal_newlines=True)
        for line in proc.stdout:
            (key, _, value) = line.partition("=")
            if key not in envs:
                continue
            os.environ[key] = value.replace("\n", '')
        
        proc.communicate()
        os.environ["TIDL_TOOLS_PATH"] = os.path.abspath(TIDL_EXTRACTED_TOOLS_PATH)
        os.environ["LD_LIBRARY_PATH"] = os.environ["TIDL_TOOLS_PATH"]
        
        if args.runtime and len(args.runtime):
            rts = args.runtime
        else:
            rts = ALL_RUNTIMES

        for rt in rts:
            log_path = os.path.join(REPORT_PATH, rt, DEVICE, OPERATORS[0])
            script = [
                "bash",
                "../run_test.sh",
                "--test_suite=operator",
                f"--tests={OPERATORS[0]}",
                f"--runtime={rt}"
            ]
            process = subprocess.run(script + ["--run_infer=0"], text=True)
            if process.returncode != 0:
                exit()
            
            for file in os.listdir(LOGS_PATH):
                if file.endswith(".html"):
                    if not os.path.exists(log_path):
                        os.makedirs(log_path)
                    shutil.copy(os.path.join(LOGS_PATH, file),  f"{log_path}/compile_without_nc.html")
                    os.remove(os.path.join(LOGS_PATH, file))
                    
            process = subprocess.run(script + ["--run_compile=0"], text=True)
            if process.returncode != 0:
                exit()
            for file in os.listdir(LOGS_PATH):
                if file.endswith(".html"):
                    shutil.copy(os.path.join(LOGS_PATH, file),  f"{log_path}/infer_ref_without_nc.html")
                    os.remove(os.path.join(LOGS_PATH, file))
            
        try:
            for env in envs:
                del os.environ[env]
            del os.environ["LD_LIBRARY_PATH"]
            del os.environ["TIDL_TOOLS_PATH"]
        except:
            pass
    else:
        script = [
            'bash',
            'run_operator_test.sh',
            f'--SOC={DEVICE}',
            '--compile_without_nc=1',   # Currently, only need to run without nc tests
            '--compile_with_nc=0',
            f'--tidl_tools_path={TIDL_TOOLS_PATH}'
        ]
        if args.operator and len(args.operator):
            script += [f'--operators={" ".join(args.operator)}']

        if args.runtime and len(args.runtime):
            script += [f'--runtimes={" ".join(args.runtime)}']
        else:
            script += [f'--runtimes={" ".join(ALL_RUNTIMES)}']
        
        process = subprocess.run(script, text=True)
        if process.returncode == 1:
            exit()

# Extract data from reports
print("──────────────────────────── Extracting data from reports ────────────────────────────")
for operator in OPERATORS:
    print(f"┌────────────────────────── Results for \x1b[95m{operator}\x1b[0m")
    for runtime in ALL_RUNTIMES:
        extract_report_data(runtime, operator, report[runtime])
    print("└─────────────────────────────────────────────────────────────────────────────────────────")

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
        data = dict.fromkeys(headers, "-")
        data['Operator name'] = res['name'].split('_')[0]
        data['Model Name'] = res['name']
        data['ONNX Subgraphs'] = res['tidl_subgraphs']
        data['ONNX Complete Offload'] = res['tidl_complete_offload']
        data['ONNXRT Functionality'] = res['outcome']
        data['ONNX Error'] = res['error']
        data['TVM Subgraphs'] = t['tidl_subgraphs']
        data['TVM Complete Offload'] = t['tidl_complete_offload']
        data['TVM Functionality'] = t['outcome']
        data['TVM Error'] = t['error']
        data["Offload Test"] = 'Passed' if res['tidl_complete_offload'] == t['tidl_complete_offload'] else 'Failed'
        data["TVM Overall"] = 'Passed' if (data['TVM Functionality'] == 'Passed' and data["Offload Test"] == 'Passed') else 'Failed'
        data["Comments"] = t.get("comment", "-")
        rows[op].append(data)

# Individual comparison report (either inference or compilation)
print("Writing CSV Comparisions >")
for op, report in rows.items():
    path = os.path.join(OUT_DIR, f"{op}_comparison.csv")
    if os.path.exists(path):
        os.remove(path) 
    if not len(rows[op]):
        continue
    with open(path, "w") as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(rows[op])

complete = {
    "with_nc": [],
    "without_nc": []
}

sg_sts = {
    "with_nc": {},
    "without_nc": {}
}

# Complete comparison report
print("Writing Complete Comparisions >")
for k in complete.keys():
    # Get compilation and inference results of both compilation and inference of with/without nc
    inference = rows[f"infer_ref_{k}"]
    compilation = rows[f"compile_{k}"]
    complete_rows = []
    total_sg_fails = 0
    for comp in compilation:
        infer = list(filter(lambda x: x["Model Name"] == comp["Model Name"], inference))
        if not len(infer):
            continue
        infer = infer[0]
        row = dict.fromkeys(complete_headers, "-")
        row["Operator Name"] = comp["Operator name"]
        row["Model Name"] = comp["Model Name"]
        row["ONNX Subgraphs"] = comp["ONNX Subgraphs"]
        row["ONNX Complete Offload"] = comp["ONNX Complete Offload"]
        row["ONNXRT Compilation Status"] = comp["ONNXRT Functionality"]
        row["ONNX Inference Status"] = infer["ONNXRT Functionality"]
        row["TVM Subgraphs"] = comp["TVM Subgraphs"]
        row["TVM Complete Offload"] = comp["TVM Complete Offload"]
        row["TVM Compilation Status"] = comp["TVM Functionality"]
        row["TVM Offload Test"] = comp["Offload Test"]
        row["TVM Compilation Overall"] = comp["TVM Overall"]
        row["TVM Inference Status"] = infer["TVM Functionality"]
        row["Compilation TVM Error"] = comp["TVM Error"]
        row["Inference TVM Error"] = infer["TVM Error"]

        if comp["Operator name"] in ignores_dict:
            modified = False
            for ignore in ignores_dict[comp["Operator name"]]:
                err = re.search(ignore["regex"], row["Compilation TVM Error"])
                if err is not None:
                    modified = True
                    if "message" in ignore:
                        row["Comments"] = "[IGNORED] " + ignore["message"]
                    else:
                        row["Comments"] = "[IGNORED] " + str(err[0])
                    for key in ignore["ignore"]:
                        row[key] = "Passed"
            
            # Recalculate based on the above changes
            if modified:
                if row["TVM Compilation Status"] == "Passed" and row["TVM Offload Test"] == "Passed":
                    row["TVM Compilation Overall"] = "Passed"
        
        overall = "-"
        if row["TVM Compilation Overall"] == "Passed" and row["TVM Inference Status"] == "Passed":
            overall = "Passed"
        elif row["TVM Compilation Overall"] != "Passed":
            overall = "Compilation Failure"
        elif row["TVM Inference Status"] != "Passed":
            overall = "Inference Failure"
        row["Overall"] = overall
        complete_rows.append(row)

        if comp["Operator name"] not in sg_sts[k]:
            sg_sts[k][comp["Operator name"]] = 0
        if row["TVM Offload Test"] == "Failed":
            sg_sts[k][comp["Operator name"]] += 1
            total_sg_fails += 1
        
    sg_sts[k]["Total"] = total_sg_fails

    path = os.path.join(OUT_DIR, f"{k}_comparison.csv")
    if os.path.exists(path):
        os.remove(path)
    if not len(complete_rows):
        continue
    with open(path, "w") as f:
        writer = csv.DictWriter(f, complete_headers)
        writer.writeheader()
        writer.writerows(complete_rows)

if args.summary:
    for runtime in ALL_RUNTIMES:
        subprocess.run(["python3", "report_summary_generation.py", "--reports_path", os.path.join(REPORT_PATH, runtime)])

if args.include_sg_fails:
    print("Modifying TVMRT test report summary...")

    try:
        with open(os.path.join(REPORT_PATH, "tvmrt", DEVICE, "summary_report.csv"), "r") as f:
            tvm_reader = csv.DictReader(f)
            fields = tvm_reader.fieldnames
            if "Offload Test Fails" not in fields:
                fields += ["Offload Test Fails"]
            new_rows = []
            for row in tvm_reader:
                row["Offload Test Fails"] = sg_sts["without_nc"].get(row["Operator name"], 0)
                new_rows.append(row)
    except Exception as e:
        print("Report summary not found, skipping")
        exit()
        
    with open(os.path.join(REPORT_PATH, "tvmrt", DEVICE, "summary_report.csv"), "w") as f:
        tvm_write = csv.DictWriter(f, fieldnames=fields)
        tvm_write.writeheader()
        tvm_write.writerows(new_rows)