import os
import csv
from bs4 import BeautifulSoup
import argparse
import sys

def parse_html_to_dict(file_path):
    """Parse an HTML file and extract test-result pairs into a dictionary."""
    with open(file_path, "r") as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")
    
    test_result_dict = {}
    test_nmse_dict = {}
    for row in rows[1:]:  # Skip the header row
        columns = row.find_all("td")
        if len(columns) >= 2:  # Ensure there are at least two columns
            result = columns[0].get_text(strip=True)
            test = columns[1].get_text(strip=True)
            nmse = columns[5].get_text(strip=True)
            test = test.split(']')[0] + ']'
            nmse = nmse.split(']')[0] + ']'
            test_result_dict[test] = result
            test_nmse_dict[test] = nmse
    return test_result_dict, test_nmse_dict


def compare_report(ref_report_path,test_report_path,output_path):
    if not os.path.exists(ref_report_path):
        print(f"[ERROR]: {ref_report_path} does not exist")
        return -1
    elif not os.path.isfile(ref_report_path):
        print(f"[ERROR]: {ref_report_path} is not a file")
        return -1

    if not os.path.exists(test_report_path):
        print(f"[ERROR]: {test_report_path} does not exist")
        return -1
    elif not os.path.isfile(test_report_path):
        print(f"[ERROR]: {test_report_path} is not a file")
        return -1

    # Parse both HTML files
    ref_dict, ref_nmse_dict = parse_html_to_dict(ref_report_path)
    test_dict, test_nmse_dict = parse_html_to_dict(test_report_path)

    diff_tests = []
    nmse_diff_tests = []
    for test, result in test_dict.items():
        if result != "Passed" and test in ref_dict and ref_dict[test] == "Passed":
            diff_tests.append(test)

    for test, nmse in test_nmse_dict.items():
        if test in ref_nmse_dict and ref_nmse_dict[test] != test_nmse_dict[test]:
            nmse_diff_tests.append(test)
    
    if (len(diff_tests) == 0) and (len(nmse_diff_tests) == 0):
        return 0

    with open(output_path,'w+') as f:
        f.write(f"TEST discrepancies: \n")
        for test in diff_tests:
            f.write(f"{test}\n")

        if (len(nmse_diff_tests) > 0):
            f.write(f"\n OUTPUT METRICS discrepancies: \n")
            for test in nmse_diff_tests:
                f.write(f"{test}\n")

    return len(diff_tests)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--ref_report_path', help='Path to pytest reference html report', type=str, required=True)
    parser.add_argument('--test_report_path', help='Path to pytest test html report', type=str, required=True)
    parser.add_argument('--output_path', help='Path to save comparison report to', type=str, default="./compare_report.txt")
    args = parser.parse_args()

    total = compare_report(args.ref_report_path, args.test_report_path, args.output_path)
    if total > 0:
        print(f"[INFO] {total} discrepancies found. Saving to {args.output_path}")
    elif total == 0:
        print(f"[INFO] Could not find any discrepancies")
    else:
        print(f"[ERROR] Something went wrong")
    
