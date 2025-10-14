#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys
import argparse
import shutil
import pandas as pd

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--reports_path', help='Path to pytest html test reports (runtimewise)', type=str, required=True)
args = parser.parse_args()

reports_dir     = args.reports_path
configs_dir  = "../../tidl_unit_test_data/configs"
out_dir      = "customer_test_reports"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

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
        if status.lower() == "error":
            status = "Failed"

        # extract the bracketed model name
        if "[" in test_cell and "]" in test_cell:
            model_name = test_cell.split("[",1)[1].split("]",1)[0]
        else:
            model_name = test_cell

        results[model_name] = (status, subgraphs, offload)
    return results

VARIANTS = [
    ("infer_ref_with_nc",  "Host Inference"),
    ("infer_ci_with_nc",   "Target Inference"),
]

operator_summaries = []

for soc in sorted(os.listdir(reports_dir)):
    soc_dir = os.path.join(reports_dir, soc)

    for op in sorted(os.listdir(soc_dir)):
        op_dir = os.path.join(soc_dir, op)
        if not os.path.isdir(op_dir):
            continue

        mod_num = None
        if '_' in op:
            mod_num = op.split('_')[-1]
            op = "_".join(op.split('_')[:-1])

        data = {k: parse_html_report(os.path.join(op_dir, f"{k}.html")) for k,_ in VARIANTS}

        # collect all model names from parsed HTML reports
        models = set().union(*[d.keys() for d in data.values()], *[])

        out_path = os.path.join(out_dir, f"{op}.csv")
        header = ["Model Name"]
        header += ["TIDL Offload"]
        for _, label in VARIANTS:
            header += [f"{label}"]

        op_cfg_dir = os.path.join(configs_dir, op)

        model_attrs = {}
        all_attr_keys = []

        # Only read configs if directory exists
        if os.path.isdir(op_cfg_dir):
            normalized_to_original = {}  # Mapping of normalized keys to original keys
            for fn in os.listdir(op_cfg_dir):
                if not fn.endswith(".csv"):
                    continue
                path = os.path.join(op_cfg_dir, fn)
                with open(path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        model_name = list(row.values())[0]
                        if mod_num and f"{op}_{mod_num}" != model_name:
                            continue
                        if model_name not in model_attrs:
                            model_attrs[model_name] = {}
                        for i, (k,v) in enumerate(row.items()):
                            if i == 0:
                                continue
                            normalized_key = k.lower().replace(" ", "_")
                            if normalized_key == "onnx_file":
                                continue
                            normalized_to_original[normalized_key] = k
                            model_attrs[model_name][normalized_key] = v
                            if normalized_key not in all_attr_keys:
                                all_attr_keys.append(normalized_key)
            
            header += [normalized_to_original[key] for key in all_attr_keys]
        else:
            # config dir missing - skip config columns
            model_attrs = {}
            all_attr_keys = []

        num_offload = 0
        total_tests = 0

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            # If configs present, use models from model_attrs, else from parsed data
            if model_attrs:
                model_list = model_attrs.keys()
            else:
                model_list = models

            for model in model_list:
                attrs = model_attrs.get(model, {})
                total_tests += 1
                row = [model]

                # Find offload status from any variant available
                offload_val = "-"
                for key, _ in VARIANTS:
                    if model in data[key]:
                        offload_val = data[key][model][2].lower()
                        break

                if offload_val == "true":
                    num_offload += 1
                    row.append("True")
                else:
                    row.append("False")

                for key, _ in VARIANTS:
                    status, subg, offload = data[key].get(model, ("N/A", "", ""))
                    row.append(status)

                for k in all_attr_keys:
                    row.append(attrs.get(k, ""))

                w.writerow(row)

        op_summary = {"Operator": op, "Total Tests": total_tests, "TIDL Offloads": num_offload}
        total_val = 0
        passed_categories = ["passed", "xpassed", "skipped", "xfailed"]
        for key, label in VARIANTS:
            da = data[key]
            if da:
                p = sum(1 for s,__,___ in da.values() if s.lower() in passed_categories)
                f = sum(1 for s,__,___ in da.values() if s.lower() not in passed_categories)
                op_summary[f"{label} Pass"] = str(p)
                op_summary[f"{label} Fail"] = str(f)
                total_val = p + f
            else:
                op_summary[f"{label} Pass"] = "-"
                op_summary[f"{label} Fail"] = "-"

        operator_summaries.append(op_summary)
        print(f"Generated {out_path}")

summary_report = os.path.join(out_dir, "Summary.csv")
fields = ["Operator", "Total Tests", "TIDL Offloads"]
for key, label in VARIANTS:
    fields.append(f"{label} Pass")
    fields.append(f"{label} Fail")

with open(summary_report, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in sorted(operator_summaries, key=lambda x: x["Operator"]):
        w.writerow(rec)

print(f"Generated Summary of all operators → {summary_report}")

# --- COMBINE CSV FILES INTO EXCEL ---
def combine_csv_to_excel(reports_dir, output_dir):
    """
    Combine all CSV files in the reports directory into a single Excel file
    """
    from openpyxl.styles import Font
    
    output_excel_file = os.path.join(output_dir, "combined_customer_report.xlsx")
    
    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
        # First, collect all CSV files and sort them to ensure Summary sheet comes first
        csv_files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]
        
        # Sort files to put Summary first
        csv_files.sort(key=lambda x: (x != 'Summary.csv', x))
        
        # Iterate over all CSV files in the directory
        for csv_file in csv_files:
            # Full path to the CSV file
            csv_path = os.path.join(reports_dir, csv_file)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)
            
            # Remove Dataset Number column if it exists
            if 'Dataset Number' in df.columns:
                df = df.drop('Dataset Number', axis=1)
            
            # Remove file paths from columns that contain "file" in their name
            for col in df.columns:
                if 'file' in col.lower():  # Check for "file" in column name
                    df = df.drop(col, axis=1)
            
            # Use the CSV file name (without extension) as the sheet name
            sheet_name = os.path.splitext(csv_file)[0]
            
            # Write the DataFrame to the Excel file as a new sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get the worksheet and apply bold formatting to header row
            worksheet = writer.sheets[sheet_name]
            bold_font = Font(bold=True)
            
            # Apply bold formatting to the first row (header row)
            for cell in worksheet[1]:  # Row 1 is the header row
                cell.font = bold_font
    
    print(f"All CSV files have been combined into {output_excel_file}")
    return output_excel_file

# Combine all generated CSV files into Excel
combined_report_path = "combined_report"
if os.path.isdir(combined_report_path):
    shutil.rmtree(combined_report_path)
os.makedirs(combined_report_path, exist_ok=True)
combine_csv_to_excel(out_dir, combined_report_path)
