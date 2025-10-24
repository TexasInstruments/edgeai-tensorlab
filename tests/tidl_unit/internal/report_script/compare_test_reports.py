#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys
import shutil
import argparse

# Import the compare_report function from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_report import compare_report

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--test_reports_path', help='Path to pytest html test reports', type=str, required=True)
parser.add_argument('--golden_reports_path', help='Path to pytest html golden reports', type=str, required=True)
args = parser.parse_args()

test_dir     = args.test_reports_path
golden_dir   = args.golden_reports_path

out_dir      = "comparison_test_reports"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

def countCompleteOffload(report_html_path):
    """Count the number of True values in the 'Complete TIDL Offload' column"""
    if not os.path.exists(report_html_path):
        return 0

    # Load the HTML file
    with open(report_html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the table with id="results-table"
    table = soup.find("table", {"id": "results-table"})
    if not table:
        return 0

    # Get the header row and find the index of "Complete TIDL Offload"
    headers = table.find("thead").find_all("th")
    tidl_offload_index = None
    for i, header in enumerate(headers):
        if header.text.strip() == "Complete TIDL Offload":
            tidl_offload_index = i
            break

    if tidl_offload_index is None:
        return 0

    # Count the number of True values in the "Complete TIDL Offload" column
    true_count = 0
    rows = table.find_all("tbody")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > tidl_offload_index:
            cell_value = cells[tidl_offload_index].text.strip()
            if cell_value == "True":
                true_count += 1

    return true_count

def extract_test_results(html_path):
    """Extract test results from HTML report"""
    if not os.path.exists(html_path):
        return (0, 0)

    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the total and passed test count
    summary_h2 = soup.find("h2", string="Summary")
    if not summary_h2:
        return (0, 0)

    summary_text = summary_h2.find_next("p").text
    total_tests = int(summary_text.split()[0])  # Extracting total test count

    passed_span  = soup.find("span", class_="passed")
    skipped_span = soup.find("span", class_="skipped")
    failed_span  = soup.find("span", class_="failed")
    error_span   = soup.find("span", class_="error")
    xfailed_span = soup.find("span", class_="xfailed")
    xpassed_span = soup.find("span", class_="xpassed")

    passed_tests = int(passed_span.text.split()[0]) if passed_span else 0
    skipped_tests = int(skipped_span.text.split()[0]) if skipped_span else 0
    failed_tests = int(failed_span.text.split()[0]) if failed_span else 0
    error_tests = int(error_span.text.split()[0]) if error_span else 0
    xfailed_tests = int(xfailed_span.text.split()[0]) if xfailed_span else 0
    xpassed_tests = int(xpassed_span.text.split()[0]) if xpassed_span else 0

    total_tests  = passed_tests + skipped_tests + failed_tests + error_tests + xfailed_tests + xpassed_tests
    total_passed = passed_tests + skipped_tests + xpassed_tests + xfailed_tests

    return total_passed, total_tests

def parse_html_report(html_path):
    """
    Returns a dict mapping model_name -> (status, subgraphs, offload)
    """
    results = {}
    if not os.path.exists(html_path):
        return results

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    rows = soup.select("table#results-table > tbody.results-table-row")
    for row in rows:
        status    = row.select_one("td.col-result").get_text(strip=True)
        test_cell = row.select_one("td.col-name").get_text(strip=True)
        subgraphs = row.select_one("td.col-name + td").get_text(strip=True)
        offload = row.select_one("td.col-name + td + td").get_text(strip=True)
        if offload == "-":
            offload = "False"
        # extract the bracketed model name
        if "[" in test_cell and "]" in test_cell:
            mn = test_cell.split("[",1)[1].split("]",1)[0]
        else:
            mn = test_cell

        results[mn] = (status, subgraphs, offload)
    return results

# Check if test and golden directories exist
if not os.path.exists(test_dir):
    print(f"[ERROR]: {test_dir} does not exist")
    sys.exit(-1)
elif not os.path.isdir(test_dir):
    print(f"[ERROR]: {test_dir} is not a directory")
    sys.exit(-1)

if not os.path.exists(golden_dir):
    print(f"[ERROR]: {golden_dir} does not exist")
    sys.exit(-1)
elif not os.path.isdir(golden_dir):
    print(f"[ERROR]: {golden_dir} is not a directory")
    sys.exit(-1)

# Find SOC directories - check all subdirectories, not just specific ones
SOC_DIRS_TEST = []
SOC_DIRS_GOLDEN = []

def is_soc_level(directory_path):
    """Check if directory contains SOC directories (AM62A, AM68A, etc.) or operator directories"""
    if not os.path.exists(directory_path):
        return False

    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    soc_names = ["AM62A", "AM67A", "AM68A", "AM69A", "TDA4VM"]

    # If any subdirectory is a known SOC name, this is SOC level
    return any(subdir in soc_names for subdir in subdirs)

def is_operator_level(directory_path):
    """Check if directory contains operator directories (by checking for HTML files in subdirectories)"""
    if not os.path.exists(directory_path):
        return False

    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

    # Check if any subdirectory contains HTML files (indicating it's an operator directory)
    for subdir in subdirs:
        subdir_path = os.path.join(directory_path, subdir)
        if any(f.endswith('.html') for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))):
            return True

    return False

# Determine the structure of input directories
test_has_socs = is_soc_level(test_dir)
golden_has_socs = is_soc_level(golden_dir)
test_has_operators = is_operator_level(test_dir)
golden_has_operators = is_operator_level(golden_dir)

print(f"Test directory structure - Has SOCs: {test_has_socs}, Has Operators: {test_has_operators}")
print(f"Golden directory structure - Has SOCs: {golden_has_socs}, Has Operators: {golden_has_operators}")

# Get all directories in test_dir and golden_dir
test_soc_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
golden_soc_dirs = [d for d in os.listdir(golden_dir) if os.path.isdir(os.path.join(golden_dir, d))]

# Handle both scenarios: SOC-level paths and operator-level paths
if test_has_socs and golden_has_socs:
    # Both paths contain SOC directories - compare matching SOCs
    print("Both paths contain SOC directories - comparing matching SOCs")
    common_socs = set(test_soc_dirs).intersection(set(golden_soc_dirs))

    for soc in common_socs:
        test_soc_path = os.path.join(test_dir, soc)
        golden_soc_path = os.path.join(golden_dir, soc)
        SOC_DIRS_TEST.append(test_soc_path)
        SOC_DIRS_GOLDEN.append(golden_soc_path)

elif test_has_operators and golden_has_operators:
    # Both paths are already at SOC level (contain operators directly)
    print("Both paths are at SOC level - comparing operators directly")
    SOC_DIRS_TEST.append(test_dir)
    SOC_DIRS_GOLDEN.append(golden_dir)

else:
    print(f"[ERROR]: Incompatible directory structures")
    print(f"Test directory - Has SOCs: {test_has_socs}, Has Operators: {test_has_operators}")
    print(f"Golden directory - Has SOCs: {golden_has_socs}, Has Operators: {golden_has_operators}")
    print(f"Both directories must have the same structure (either both contain SOCs or both contain operators)")
    sys.exit(-1)

if len(SOC_DIRS_TEST) == 0:
    print(f"[ERROR]: Could not find any matching directories to compare")
    print(f"Test directories: {test_soc_dirs}")
    print(f"Golden directories: {golden_soc_dirs}")
    sys.exit(-1)

for soc_test_dir, soc_golden_dir in zip(SOC_DIRS_TEST, SOC_DIRS_GOLDEN):
    soc_name = os.path.basename(soc_test_dir)

    # Handle case where we're comparing at operator level (no SOC subdirectories)
    if soc_test_dir != test_dir:
        print(f"Processing SOC: {soc_name}")

    # Create SOC-specific output directory
    soc_out_dir = os.path.join(out_dir, soc_name)
    os.makedirs(soc_out_dir, exist_ok=True)

    # Get list of operators (folders) in both test and golden directories
    test_operators = set([folder for folder in os.listdir(soc_test_dir) if os.path.isdir(os.path.join(soc_test_dir, folder))])
    golden_operators = set([folder for folder in os.listdir(soc_golden_dir) if os.path.isdir(os.path.join(soc_golden_dir, folder))])

    # Find common operators
    common_operators = test_operators.intersection(golden_operators)

    if len(common_operators) == 0:
        print(f"No common operators found between test and golden directories for SOC {soc_name}")
        continue

    # Discover all unique variants across all operators by scanning golden path
    all_variants = set()
    operator_variants = {}

    for op in sorted(common_operators):
        op_golden_path = os.path.join(soc_golden_dir, op)
        op_test_path = os.path.join(soc_test_dir, op)

        # Get all HTML files in golden operator path
        variants = []
        if os.path.isdir(op_golden_path):
            for file in os.listdir(op_golden_path):
                if file.endswith('.html'):
                    variant_name = file.replace('.html', '')
                    # Check if corresponding file exists in test path
                    test_file_path = os.path.join(op_test_path, file)
                    if os.path.exists(test_file_path):
                        # Create a readable label from the variant name
                        variant_label = variant_name.replace('_', ' ').title()
                        variants.append((variant_name, variant_label))

        operator_variants[op] = variants
        all_variants.update(variants)

    if len(all_variants) == 0:
        print(f"No HTML report variants found for SOC {soc_name}")
        continue

    print(f"Found variants for {soc_name}: {[v[0] for v in all_variants]}")

    # Prepare the CSV headers similar to report_summary_generation.py
    headers = ["Operator name", "Total Test", "TIDL Offload Percentage Test", "TIDL Offload Percentage Golden"]

    # Add headers for each variant
    for variant_name, variant_label in all_variants:
        headers.extend([f"{variant_label} Test Failures", f"{variant_label} Golden Failures"])

    # Collect results
    rows = []

    total = dict.fromkeys(headers, 0)
    total["Operator name"] = "Total"
    total_offload_test = 0
    total_offload_golden = 0
    total_tests_count = 0

    for op in sorted(common_operators):
        op_test_path = os.path.join(soc_test_dir, op)
        op_golden_path = os.path.join(soc_golden_dir, op)

        # Get variants for this specific operator
        variants = operator_variants[op]

        if len(variants) == 0:
            print(f"Skipping {op} as no matching HTML reports found")
            continue

        # parse both sets - only for variants available for this operator
        data_test = {}
        data_golden = {}
        for variant_name, variant_label in variants:
            data_test[variant_name] = parse_html_report(os.path.join(op_test_path, f"{variant_name}.html"))
            data_golden[variant_name] = parse_html_report(os.path.join(op_golden_path, f"{variant_name}.html"))

        # Initialize variables for this operator
        total_tests = "N/A"
        offloaded_tests_test = "N/A"
        offloaded_tests_golden = "N/A"

        # Dictionary to store failure counts for each variant
        variant_failures = {}

        # Process each variant available for this operator
        for variant_name, variant_label in variants:
            test_report_path = os.path.join(op_test_path, f"{variant_name}.html")
            golden_report_path = os.path.join(op_golden_path, f"{variant_name}.html")

            test_failures = "N/A"
            golden_failures = "N/A"

            if os.path.exists(test_report_path):
                test_results = extract_test_results(test_report_path)
                if offloaded_tests_test == "N/A":
                    offloaded_tests_test = countCompleteOffload(test_report_path)
                if total_tests == "N/A":
                    total_tests = test_results[1]
                test_failures = total_tests - test_results[0]

            if os.path.exists(golden_report_path):
                golden_results = extract_test_results(golden_report_path)
                if offloaded_tests_golden == "N/A":
                    offloaded_tests_golden = countCompleteOffload(golden_report_path)
                if total_tests == "N/A":
                    total_tests = golden_results[1]
                golden_failures = total_tests - golden_results[0]

            variant_failures[variant_name] = (test_failures, golden_failures)

            # Generate comparison report files similar to report_summary_generation.py
            if os.path.exists(test_report_path) and os.path.exists(golden_report_path):
                # Create operator-specific directory
                op_out_dir = os.path.join(soc_out_dir, op)
                os.makedirs(op_out_dir, exist_ok=True)

                compare_report_path = os.path.join(op_out_dir, f"{variant_name}_golden_vs_test.txt")
                compare_result = compare_report(golden_report_path, test_report_path, output_path=compare_report_path)

        # Calculate offload percentages
        if offloaded_tests_test == "N/A" or total_tests == "N/A":
            offloaded_tests_percentage_test = "N/A"
        else:
            offloaded_tests_percentage_test = ((float(offloaded_tests_test) / float(total_tests) * 100))

        if offloaded_tests_golden == "N/A" or total_tests == "N/A":
            offloaded_tests_percentage_golden = "N/A"
        else:
            offloaded_tests_percentage_golden = ((float(offloaded_tests_golden) / float(total_tests) * 100))

        # Build row for this operator
        row = [op, str(total_tests), str(offloaded_tests_percentage_test), str(offloaded_tests_percentage_golden)]

        # Add failure counts for each variant (all variants, not just those available for this operator)
        for variant_name, variant_label in all_variants:
            test_failures, golden_failures = variant_failures.get(variant_name, ("N/A", "N/A"))
            row.extend([str(test_failures), str(golden_failures)])

        rows.append(row)

        # Update totals
        if total_tests != "N/A":
            total_tests_count += total_tests
            total["Total Test"] += total_tests
        if offloaded_tests_test != "N/A":
            total_offload_test += offloaded_tests_test
        if offloaded_tests_golden != "N/A":
            total_offload_golden += offloaded_tests_golden

        # Update failure totals
        for variant_name, variant_label in all_variants:
            test_failures, golden_failures = variant_failures.get(variant_name, ("N/A", "N/A"))
            if test_failures != "N/A":
                total[f"{variant_label} Test Failures"] += test_failures
            if golden_failures != "N/A":
                total[f"{variant_label} Golden Failures"] += golden_failures

        print(f"Processed operator: {op}")

    # Calculate total percentages
    if total_tests_count > 0:
        total["TIDL Offload Percentage Test"] = (float(total_offload_test) / float(total_tests_count) * 100)
        total["TIDL Offload Percentage Golden"] = (float(total_offload_golden) / float(total_tests_count) * 100)
    else:
        total["TIDL Offload Percentage Test"] = 0.0
        total["TIDL Offload Percentage Golden"] = 0.0

    # Add total row
    total_row = [total["Operator name"], str(total["Total Test"]),
                 str(total["TIDL Offload Percentage Test"]), str(total["TIDL Offload Percentage Golden"])]

    for variant_name, variant_label in all_variants:
        total_row.extend([str(total[f"{variant_label} Test Failures"]), str(total[f"{variant_label} Golden Failures"])])

    rows.append(total_row)

    # Write results to CSV file
    output_csv = os.path.join(soc_out_dir, "comparison_summary_report.csv")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header
        writer.writerows(rows)  # Write data rows

    print(f"Summary report saved to {output_csv}")

    # Generate individual operator CSV files with detailed model comparisons
    for op in sorted(common_operators):
        op_test_path = os.path.join(soc_test_dir, op)
        op_golden_path = os.path.join(soc_golden_dir, op)

        # Get variants for this specific operator
        variants = operator_variants[op]

        if len(variants) == 0:
            continue

        # parse both sets - only for variants available for this operator
        data_test = {}
        data_golden = {}
        for variant_name, variant_label in variants:
            data_test[variant_name] = parse_html_report(os.path.join(op_test_path, f"{variant_name}.html"))
            data_golden[variant_name] = parse_html_report(os.path.join(op_golden_path, f"{variant_name}.html"))

        # collect all model names
        models = set().union(*[d.keys() for d in data_test.values()], *[d.keys() for d in data_golden.values()])

        if len(models) == 0:
            continue

        # write per-operator CSV
        out_path = os.path.join(soc_out_dir, f"{op}.csv")
        header = ["Model name", "Status", "TIDL_Offload_test", "TIDL_Offload_golden"]

        # Add headers for variants available for this operator
        for variant_name, variant_label in variants:
            col = variant_label.replace(" ", "_")
            header += [f"{col}_test", f"{col}_golden"]

        # No config attributes needed

        num_upgraded = 0
        num_degraded = 0
        num_offload = 0
        total_tests = 0

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            # model-by-model rows - sort models numerically
            def natural_sort_key(model_name):
                """Sort key function for natural sorting (handles numbers properly)"""
                import re
                parts = re.split(r'(\d+)', model_name)
                return [int(part) if part.isdigit() else part for part in parts]

            for mn in sorted(models, key=natural_sort_key):
                row = [mn]
                total_tests += 1
                offload_test = "N/A"
                offload_golden = "N/A"
                degraded = False
                upgraded = False

                # Check status across all variants to determine overall status
                for variant_name, variant_label in variants:
                    status_test, subgraphs, offload_test_variant = data_test[variant_name].get(mn, ("N/A", "N/A", "N/A"))
                    status_golden, subgraphs, offload_golden_variant = data_golden[variant_name].get(mn, ("N/A", "N/A", "N/A"))

                    if offload_test == "N/A" and offload_test_variant != "N/A":
                        offload_test = offload_test_variant
                    if offload_golden == "N/A" and offload_golden_variant != "N/A":
                        offload_golden = offload_golden_variant

                    if status_test != "N/A" and status_golden != "N/A":
                        if status_test.lower() == "passed" and status_golden.lower() == "failed":
                            upgraded = True
                        elif status_test.lower() == "failed" and status_golden.lower() == "passed":
                            degraded = True

                # Determine overall status
                if degraded == False and upgraded == False:
                    row += ["Same"]
                elif degraded == True:
                    row += ["Degraded"]
                    num_degraded += 1
                elif upgraded == True:
                    row += ["Upgraded"]
                    num_upgraded += 1
                else:
                    row += ["N/A"]

                row += [offload_test, offload_golden]
                if offload_test.lower() == "true":
                    num_offload += 1

                # Add status for each variant available for this operator
                for variant_name, variant_label in variants:
                    status_test, subgraphs, offload_test_variant = data_test[variant_name].get(mn, ("N/A", "N/A", "N/A"))
                    status_golden, subgraphs, offload_golden_variant = data_golden[variant_name].get(mn, ("N/A", "N/A", "N/A"))
                    row += [status_test, status_golden]

                w.writerow(row)

        print(f"Generated detailed operator report: {out_path}")

print(f"Comparison reports generated in: {out_dir}")
