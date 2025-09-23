#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys
import argparse
import shutil

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

        op_summary = {"Operator": op, "Total_Tests": total_tests, "TIDL_Offload_Percentage": num_offload}
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

        # Avoid division by zero
        if total_tests > 0:
            op_summary["TIDL_Offload_Percentage"] = (num_offload / total_tests) * 100
        else:
            op_summary["TIDL_Offload_Percentage"] = 0

        operator_summaries.append(op_summary)
        print(f"Wrote {out_path}")

# --- FULL OPERATOR COMPARISON ---
full_path = os.path.join(out_dir, "operator_test_report_summary.csv")
fields = ["Operator", "Total_Tests", "TIDL_Offload_Percentage"]
for key, label in VARIANTS:
    fields.append(f"{label} Pass")
    fields.append(f"{label} Fail")

with open(full_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in sorted(operator_summaries, key=lambda x: x["Operator"]):
        w.writerow(rec)

print(f"Wrote consolidated comparison → {full_path}")