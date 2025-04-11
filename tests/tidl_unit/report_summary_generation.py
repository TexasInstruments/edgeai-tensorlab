import os
import csv
from bs4 import BeautifulSoup

# Define the directory containing all report folders
REPORTS_DIR = "operator_test_reports"

# Output CSV file
OUTPUT_CSV = f"{REPORTS_DIR}/summary_report.csv"

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

# Prepare the CSV headers
headers = ["Operator name", "Total Test", "TIDL RT (Compile-NO_NC)", "TIDL RT (Inference-HE-No NC)", "TIDL RT (Inference-HE-with NC)", "TIDL Offload Percentage"]
# Collect results
rows = []
# Iterate over each folder in the reports directory
folders = sorted([folder for folder in os.listdir(REPORTS_DIR) if os.path.isdir(os.path.join(REPORTS_DIR, folder))])
for folder in folders:
    folder_path = os.path.join(REPORTS_DIR, folder)
    if os.path.isdir(folder_path):
        # Paths to the expected report files
        compile_without_nc_path = os.path.join(folder_path, "compile_without_nc.html")
        compile_with_nc_path = os.path.join(folder_path, "compile_with_nc.html")
        infer_without_nc_path = os.path.join(folder_path, "infer_without_nc.html")
        infer_with_nc_path = os.path.join(folder_path, "infer_with_nc.html")

        # Extract test results
        compile_without_nc_results = extract_test_results(compile_without_nc_path) if os.path.exists(compile_without_nc_path) else ("N/A","N/A")
        infer_without_nc_results = extract_test_results(infer_without_nc_path) if os.path.exists(infer_without_nc_path) else ("N/A","N/A")
        infer_with_nc_results = extract_test_results(infer_with_nc_path) if os.path.exists(infer_with_nc_path) else ("N/A","N/A")
        offLoadedTests = countCompleteOffload(compile_with_nc_path) if os.path.exists(compile_with_nc_path) else 0

        # Append row to results
        t = compile_without_nc_results[1]
        a = compile_without_nc_results[0]
        b = infer_without_nc_results[0]
        c = infer_with_nc_results[0]

        rows.append([folder, str(t), "N/A" if (a=="N/A") else str(int(t)-int(a)), "N/A" if (b=="N/A") else str(int(t)-int(b)), "N/A" if (c=="N/A") else str(int(t)-int(c)), "N/A" if (t=="N/A") else str((offLoadedTests / t) * 100)])

# Write results to a CSV file
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write header
    writer.writerows(rows)  # Write data rows

print(f"Summary report saved to {OUTPUT_CSV}")