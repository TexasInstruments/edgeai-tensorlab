import os
import csv
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--reports_path', help='Path to pytest html reports', type=str, required=True)
args = parser.parse_args()

def countCompleteOffload(report_html_path):
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
    true_count = 0
    # rows = table.find("tbody").find_all("tbody")
    rows = table.find_all("tbody")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > tidl_offload_index:
            cell_value = cells[tidl_offload_index].text.strip()
            if cell_value == "True":
                true_count += 1

    return true_count

def extract_test_results(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the total and passed test count
    summary_text = soup.find("h2", string="Summary").find_next("p").text
    total_tests = int(summary_text.split()[0])  # Extracting total test count
    passed_span = soup.find("span", class_="passed")
    passed_tests = int(passed_span.text.split()[0]) if passed_span else 0
    return passed_tests,total_tests

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
    drop_compile_without_nc=True
    drop_compile_with_nc=True
    drop_infer_ref_without_nc=True
    drop_infer_ref_with_nc=True
    drop_infer_natc_with_nc=True
    drop_infer_ci_with_nc=True
    drop_infer_target_with_nc=True

    # Prepare the CSV headers
    headers = ["Operator name", "Total Test", "TIDL Offload Percentage", "Compile without NC", "Compile with NC", "Infer REF without NC", "Infer REF with NC", "Infer NATC with NC", "Infer CI with NC", "Infer TARGET with NC"]
    # Collect results
    rows = []
    # Iterate over each folder in the reports directory
    folders = sorted([folder for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder))])
    for folder in folders:
        folder_path = os.path.join(dir, folder)
        if os.path.isdir(folder_path):
            # Paths to the expected report files
            compile_without_nc_path = os.path.join(folder_path, "compile_without_nc.html")
            compile_with_nc_path = os.path.join(folder_path, "compile_with_nc.html")
            infer_ref_without_nc_path = os.path.join(folder_path, "infer_ref_without_nc.html")
            infer_ref_with_nc_path = os.path.join(folder_path, "infer_ref_with_nc.html")
            infer_natc_with_nc_path = os.path.join(folder_path, "infer_natc_with_nc.html")
            infer_ci_with_nc_path = os.path.join(folder_path, "infer_ci_with_nc.html")
            infer_target_with_nc_path = os.path.join(folder_path, "infer_target_with_nc.html")

            total_tests = "N/A"
            offloaded_tests = "N/A"
            compile_without_nc_failures = "N/A"
            compile_with_nc_failures = "N/A"
            infer_ref_without_nc_failures = "N/A"
            infer_ref_with_nc_failures = "N/A"
            infer_natc_with_nc_failures = "N/A"
            infer_ci_with_nc_failures = "N/A"
            infer_target_with_nc_failures = "N/A"

            if os.path.exists(compile_without_nc_path):
                compile_without_nc_results = extract_test_results(compile_without_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(compile_without_nc_path)
                if total_tests == "N/A":
                    total_tests = compile_without_nc_results[1]
                compile_without_nc_failures = total_tests - compile_without_nc_results[0]
                drop_compile_without_nc=False

            if os.path.exists(compile_with_nc_path):
                compile_with_nc_results = extract_test_results(compile_with_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(compile_with_nc_path)
                if total_tests == "N/A":
                    total_tests = compile_with_nc_results[1]
                compile_with_nc_failures = total_tests - compile_with_nc_results[0]
                drop_compile_with_nc=False


            if os.path.exists(infer_ref_without_nc_path):
                infer_ref_without_nc_results = extract_test_results(infer_ref_without_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(infer_ref_without_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_ref_without_nc_results[1]
                infer_ref_without_nc_failures = total_tests - infer_ref_without_nc_results[0]
                drop_infer_ref_without_nc=False

            if os.path.exists(infer_ref_with_nc_path):
                infer_ref_with_nc_results = extract_test_results(infer_ref_with_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(infer_ref_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_ref_with_nc_results[1]
                infer_ref_with_nc_failures = total_tests - infer_ref_with_nc_results[0]
                drop_infer_ref_with_nc=False

            if os.path.exists(infer_natc_with_nc_path):
                infer_natc_with_nc_results = extract_test_results(infer_natc_with_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(infer_natc_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_natc_with_nc_results[1]
                infer_natc_with_nc_failures = total_tests - infer_natc_with_nc_results[0]
                drop_infer_natc_with_nc=False

            if os.path.exists(infer_ci_with_nc_path):
                infer_ci_with_nc_results = extract_test_results(infer_ci_with_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(infer_ci_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_ci_with_nc_results[1]
                infer_ci_with_nc_failures = total_tests - infer_ci_with_nc_results[0]
                drop_infer_ci_with_nc=False

            if os.path.exists(infer_target_with_nc_path):
                infer_target_with_nc_results = extract_test_results(infer_target_with_nc_path)
                if offloaded_tests == "N/A":
                    offloaded_tests = countCompleteOffload(infer_target_with_nc_path)
                if total_tests == "N/A":
                    total_tests = infer_target_with_nc_results[1]
                infer_target_with_nc_failures = total_tests - infer_target_with_nc_results[0]
                drop_infer_target_with_nc=False

            if offloaded_tests == "N/A" or total_tests == "N/A":
                offloaded_tests_percentage = "N/A"
            else:
                offloaded_tests_percentage = ((float(offloaded_tests) / float(total_tests) * 100))

            rows.append([folder, str(total_tests), str(offloaded_tests_percentage),
                        str(compile_without_nc_failures), str(compile_with_nc_failures),
                        str(infer_ref_without_nc_failures), str(infer_ref_with_nc_failures),
                        str(infer_natc_with_nc_failures), str(infer_ci_with_nc_failures),
                        str(infer_target_with_nc_failures)])

    drop_idx = []
    if drop_compile_without_nc:
        drop_idx.append(3)
    if drop_compile_with_nc:
        drop_idx.append(4)
    if drop_infer_ref_without_nc:
        drop_idx.append(5)
    if drop_infer_ref_with_nc:
        drop_idx.append(6)
    if drop_infer_natc_with_nc:
        drop_idx.append(7)
    if drop_infer_ci_with_nc:
        drop_idx.append(8)
    if drop_infer_target_with_nc:
        drop_idx.append(9)
    
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