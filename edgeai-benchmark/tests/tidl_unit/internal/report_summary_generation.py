import os
import csv
from bs4 import BeautifulSoup
import argparse
import sys
from compare_report import compare_report

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--reports_path', help='Path to pytest html reports', type=str, required=True)
args = parser.parse_args()

def countOffloads(report_html_path):
    # Load the HTML file
    with open(report_html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the table with id="results-table"
    table = soup.find("table", {"id": "results-table"})

    # Get the header row and find the index of "Complete TIDL Offload"
    headers = table.find("thead").find_all("th")
    tidl_offload_index = None
    for i, header in enumerate(headers):
        if header.text.strip() == "Complete TIDL Offload":
            tidl_offload_index = i
            break

    if tidl_offload_index is None:
        print("Column 'Complete TIDL Offload' not found.")
        exit()

    # Count the number of True values in the "Complete TIDL Offload" column
    complete_offload = 0
    partial_offload = 0
    no_offload = 0
    rows = table.find_all("tbody")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > tidl_offload_index:
            cell_value = cells[tidl_offload_index].text.strip()
            if cell_value == "True":
                complete_offload += 1
            elif cell_value == "False":
                partial_offload += 1
            else:
                no_offload +=1 

    return complete_offload, partial_offload, no_offload

def extract_test_results(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the total and passed test count
    summary_text = soup.find("h2", string="Summary").find_next("p").text
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

if not os.path.exists(args.reports_path):
    print(f"[ERROR]: {args.reports_path} does not exist")
    sys.exit(-1)
elif not os.path.isdir(args.reports_path):
    print(f"[ERROR]: {args.reports_path} is not a directory")
    sys.exit(-1)

SOC_DIR = []
for folder in os.listdir(args.reports_path):
    if folder in ["AM62A", "AM67A", "AM68A", "AM69A", "TDA4VM"]:
        SOC_DIR.append(os.path.join(args.reports_path, folder))

if len(SOC_DIR) == 0:
    print(f"[ERROR]: Could not find any directory (AM62A, AM67A, AM68A, AM69A, TDA4VM) in {args.reports_path}")
    sys.exit(-1)

for dir in SOC_DIR:
    drop_compile_without_nc   = True
    drop_compile_with_nc      = True
    drop_infer_ref_without_nc = True
    drop_infer_ref_with_nc    = True
    drop_infer_natc_with_nc   = True
    drop_infer_ci_with_nc     = True
    drop_infer_target_with_nc = True
    drop_infer_ref            = True
    drop_infer_natc           = True
    drop_infer_ci             = True
    drop_infer_target         = True

    # Prepare the CSV headers
    headers = ["Test name", "Total Test", "Complete Offload", "Partial Offload", "No Offload", "Compile without NC Failures", "Compile with NC Failures", "Infer REF without NC Failures", "Infer REF with NC Failures", "Infer NATC with NC Failures", "Infer CI with NC Failures", "Infer TARGET with NC Failures", "Infer REF Failures", "Infer NATC Failures", "Infer CI Failures", "Infer TARGET Failures"]
    # Collect results
    rows = []
    # Iterate over each folder in the reports directory
    folders = sorted([folder for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder))])

    total = dict.fromkeys(headers, 0)
    total["Test name"] = "Total"

    for folder in folders:
        folder_path = os.path.join(dir, folder)
        compile_without_nc = False
        compile_with_nc = False
        infer_ref_without_nc = False
        infer_ref_with_nc = False
        infer_natc_with_nc = False
        infer_ci_with_nc = False
        infer_target_with_nc = False
        infer_ref = False
        infer_natc = False
        infer_ci = False
        infer_target = False

        if os.path.isdir(folder_path):
            # Paths to the expected report files
            compile_without_nc_path   = os.path.join(folder_path, "compile_without_nc.html")
            compile_with_nc_path      = os.path.join(folder_path, "compile_with_nc.html")
            infer_ref_without_nc_path = os.path.join(folder_path, "infer_ref_without_nc.html")
            infer_ref_with_nc_path    = os.path.join(folder_path, "infer_ref_with_nc.html")
            infer_natc_with_nc_path   = os.path.join(folder_path, "infer_natc_with_nc.html")
            infer_ci_with_nc_path     = os.path.join(folder_path, "infer_ci_with_nc.html")
            infer_target_with_nc_path = os.path.join(folder_path, "infer_target_with_nc.html")
            infer_ref_path            = os.path.join(folder_path, "infer_ref.html")
            infer_natc_path           = os.path.join(folder_path, "infer_natc.html")
            infer_ci_path             = os.path.join(folder_path, "infer_ci.html")
            infer_target_path         = os.path.join(folder_path, "infer_target.html")

            total_tests                   = "N/A"
            complete_offloaded_tests      = "N/A"
            partial_offloaded_tests       = "N/A"
            no_offloaded_tests            = "N/A"
            compile_without_nc_failures   = "N/A"
            compile_with_nc_failures      = "N/A"
            infer_ref_without_nc_failures = "N/A"
            infer_ref_with_nc_failures    = "N/A"
            infer_natc_with_nc_failures   = "N/A"
            infer_ci_with_nc_failures     = "N/A"
            infer_target_with_nc_failures = "N/A"
            infer_ref_failures            = "N/A"
            infer_natc_failures           = "N/A"
            infer_ci_failures             = "N/A"
            infer_target_failures         = "N/A"

            if os.path.exists(compile_without_nc_path):
                compile_without_nc_results = extract_test_results(compile_without_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(compile_without_nc_path)
                if total_tests == "N/A":
                    total_tests = compile_without_nc_results[1]
                compile_without_nc_failures = total_tests - compile_without_nc_results[0]
                drop_compile_without_nc = False
                compile_without_nc = True

            if os.path.exists(compile_with_nc_path):
                compile_with_nc_results = extract_test_results(compile_with_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(compile_with_nc_path)
                if total_tests == "N/A":
                    total_tests = compile_with_nc_results[1]
                compile_with_nc_failures = total_tests - compile_with_nc_results[0]
                drop_compile_with_nc = False
                compile_with_nc = True


            if os.path.exists(infer_ref_without_nc_path):
                infer_ref_without_nc_results = extract_test_results(infer_ref_without_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_ref_without_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_ref_without_nc_results[1]
                infer_ref_without_nc_failures = total_tests - infer_ref_without_nc_results[0]
                drop_infer_ref_without_nc = False
                infer_ref_without_nc = True

            if os.path.exists(infer_ref_with_nc_path):
                infer_ref_with_nc_results = extract_test_results(infer_ref_with_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_ref_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_ref_with_nc_results[1]
                infer_ref_with_nc_failures = total_tests - infer_ref_with_nc_results[0]
                drop_infer_ref_with_nc = False
                infer_ref_with_nc = True

            if os.path.exists(infer_natc_with_nc_path):
                infer_natc_with_nc_results = extract_test_results(infer_natc_with_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_natc_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_natc_with_nc_results[1]
                infer_natc_with_nc_failures = total_tests - infer_natc_with_nc_results[0]
                drop_infer_natc_with_nc = False
                infer_natc_with_nc = True

            if os.path.exists(infer_ci_with_nc_path):
                infer_ci_with_nc_results = extract_test_results(infer_ci_with_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_ci_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_ci_with_nc_results[1]
                infer_ci_with_nc_failures = total_tests - infer_ci_with_nc_results[0]
                drop_infer_ci_with_nc = False
                infer_ci_with_nc = True

            if os.path.exists(infer_target_with_nc_path):
                infer_target_with_nc_results = extract_test_results(infer_target_with_nc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_target_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_target_with_nc_results[1]
                infer_target_with_nc_failures = total_tests - infer_target_with_nc_results[0]
                drop_infer_target_with_nc = False
                infer_target_with_nc = True

            if os.path.exists(infer_ref_path):
                infer_ref_results = extract_test_results(infer_ref_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_ref_path)
                if total_tests == "N/A":
                    total_tests = infer_ref_results[1]
                infer_ref_failures = total_tests - infer_ref_results[0]
                drop_infer_ref = False
                infer_ref = True

            if os.path.exists(infer_natc_path):
                infer_natc_results = extract_test_results(infer_natc_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_natc_path)
                if total_tests == "N/A":
                    total_tests = infer_natc_results[1]
                infer_natc_failures = total_tests - infer_natc_results[0]
                drop_infer_natc = False
                infer_natc = True

            if os.path.exists(infer_ci_path):
                infer_ci_results = extract_test_results(infer_ci_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_ci_path)
                if total_tests == "N/A":
                    total_tests = infer_ci_results[1]
                infer_ci_failures = total_tests - infer_ci_results[0]
                drop_infer_ci = False
                infer_ci = True

            if os.path.exists(infer_target_path):
                infer_target_results = extract_test_results(infer_target_path)
                if complete_offloaded_tests == "N/A":
                    complete_offloaded_tests, partial_offloaded_tests, no_offloaded_tests = countOffloads(infer_target_path)
                if total_tests == "N/A":
                    total_tests = infer_target_results[1]
                infer_target_failures = total_tests - infer_target_results[0]
                drop_infer_target = False
                infer_target = True

            total["Total Test"]                       += total_tests if total_tests != "N/A" else 0
            total["Complete Offload"]                 += complete_offloaded_tests if complete_offloaded_tests != "N/A" else 0
            total["Partial Offload"]                  += partial_offloaded_tests if partial_offloaded_tests != "N/A" else 0
            total["No Offload"]                       += no_offloaded_tests if no_offloaded_tests != "N/A" else 0
            total["Compile without NC Failures"]      += compile_without_nc_failures if compile_without_nc_failures != "N/A" else 0
            total["Compile with NC Failures"]         += compile_with_nc_failures if compile_with_nc_failures != "N/A" else 0
            total["Infer REF without NC Failures"]    += infer_ref_without_nc_failures if infer_ref_without_nc_failures != "N/A" else 0
            total["Infer REF with NC Failures"]       += infer_ref_with_nc_failures if infer_ref_with_nc_failures != "N/A" else 0
            total["Infer NATC with NC Failures"]      += infer_natc_with_nc_failures if infer_natc_with_nc_failures != "N/A" else 0
            total["Infer CI with NC Failures"]        += infer_ci_with_nc_failures if infer_ci_with_nc_failures != "N/A" else 0
            total["Infer TARGET with NC Failures"]    += infer_target_with_nc_failures if infer_target_with_nc_failures != "N/A" else 0
            total["Infer REF Failures"]               += infer_ref_failures if infer_ref_failures != "N/A" else 0
            total["Infer NATC Failures"]              += infer_natc_failures if infer_natc_failures != "N/A" else 0
            total["Infer CI Failures"]                += infer_ci_failures if infer_ci_failures != "N/A" else 0
            total["Infer TARGET Failures"]            += infer_target_failures if infer_target_failures != "N/A" else 0

            rows.append([folder, str(total_tests), str(complete_offloaded_tests),
                        str(partial_offloaded_tests),str(no_offloaded_tests),
                        str(compile_without_nc_failures), str(compile_with_nc_failures),
                        str(infer_ref_without_nc_failures), str(infer_ref_with_nc_failures),
                        str(infer_natc_with_nc_failures), str(infer_ci_with_nc_failures),
                        str(infer_target_with_nc_failures), str(infer_ref_failures),
                        str(infer_natc_failures), str(infer_ci_failures),
                        str(infer_target_failures)])

            if infer_ref_with_nc and infer_target_with_nc:
                compare_report_path = os.path.join(folder_path, "infer_ref_vs_evm.txt")
                infer_ref_vs_evm = compare_report(infer_ref_with_nc_path, infer_target_with_nc_path, output_path=compare_report_path)
            if infer_ref_with_nc and infer_ref_without_nc:
                compare_report_path = os.path.join(folder_path, "infer_ref_no_nc_vs_ref_nc.txt")
                infer_ref_no_nc_vs_ref_nc = compare_report(infer_ref_without_nc_path, infer_ref_with_nc_path, output_path=compare_report_path)
            if infer_ref_with_nc and infer_natc_with_nc:
                compare_report_path = os.path.join(folder_path, "infer_ref_nc_vs_natc_nc.txt")
                infer_ref_nc_vs_natc_nc = compare_report(infer_ref_with_nc_path, infer_natc_with_nc_path, output_path=compare_report_path)
            if infer_natc_with_nc and infer_ci_with_nc:
                compare_report_path   = os.path.join(folder_path, "infer_natc_nc_vs_ci_nc.txt")
                infer_natc_nc_vs_ci_nc = compare_report(infer_natc_with_nc_path, infer_ci_with_nc_path, output_path=compare_report_path)
            if infer_ref and infer_natc:
                compare_report_path = os.path.join(folder_path, "infer_ref_vs_natc.txt")
                infer_ref_vs_natc = compare_report(infer_ref_path, infer_natc_path, output_path=compare_report_path)
            if infer_natc and infer_ci:
                compare_report_path = os.path.join(folder_path, "infer_natc_vs_ci.txt")
                infer_natc_vs_ci = compare_report(infer_natc_path, infer_ci_path, output_path=compare_report_path)
            if infer_ref and infer_target:
                compare_report_path   = os.path.join(folder_path, "infer_ref_vs_target.txt")
                infer_ref_vs_target = compare_report(infer_ref_path, infer_target_path, output_path=compare_report_path)

    rows.append(total.values())

    drop_idx = []
    if drop_compile_without_nc:
        drop_idx.append(5)
    if drop_compile_with_nc:
        drop_idx.append(6)
    if drop_infer_ref_without_nc:
        drop_idx.append(7)
    if drop_infer_ref_with_nc:
        drop_idx.append(8)
    if drop_infer_natc_with_nc:
        drop_idx.append(9)
    if drop_infer_ci_with_nc:
        drop_idx.append(10)
    if drop_infer_target_with_nc:
        drop_idx.append(11)
    if drop_infer_ref:
        drop_idx.append(12)
    if drop_infer_natc:
        drop_idx.append(13)
    if drop_infer_ci:
        drop_idx.append(14)
    if drop_infer_target:
        drop_idx.append(15)
    
    headers = [item for i, item in enumerate(headers) if i not in drop_idx]
    for idx, row in enumerate(rows):
        row = [item for i, item in enumerate(row) if i not in drop_idx]
        rows[idx] = row

    output_csv = os.path.join(dir, "summary_report.csv")
    # Write results to a CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header
        writer.writerows(rows)  # Write data rows

    print(f"Summary report saved to {output_csv}")