#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys
import shutil
import argparse

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--test_reports_path', help='Path to pytest html test reports', type=str, required=True)
parser.add_argument('--golden_reports_path', help='Path to pytest html goldenn reports', type=str, required=True)
args = parser.parse_args()

test_dir     = args.test_reports_path
golden_dir   = args.golden_reports_path
configs_dir  = "../../tidl_unit_test_data/configs"

out_dir      = "comparison_test_reports"
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
        # extract the bracketed model name
        if "[" in test_cell and "]" in test_cell:
            mn = test_cell.split("[",1)[1].split("]",1)[0]
        else:
            mn = test_cell

        results[mn] = (status, subgraphs, offload)
    return results


VARIANTS = [
    ("compile_with_nc",        "Compile with NC"),
    ("compile_without_nc",     "Compile without NC"),
    ("infer_ref_with_nc",      "Infer Ref with NC"),
    ("infer_ref_without_nc",   "Infer Ref without NC"),
]

# Will accumulate per-operator summary for the final CSV
operator_summaries = []

for soc in sorted(os.listdir(test_dir)):
    soc_test_dir = os.path.join(test_dir, soc)
    soc_golden_dir = os.path.join(golden_dir, soc)
    if not (os.path.isdir(soc_test_dir) and os.path.isdir(soc_golden_dir)):
        continue

    for op in sorted(os.listdir(soc_test_dir)):
        op_test_path = os.path.join(soc_test_dir, op)
        op_golden_path  = os.path.join(soc_golden_dir, op)
        if not (os.path.isdir(op_test_path) and os.path.isdir(op_golden_path)):
            continue

        # parse both sets
        data_test = {k: parse_html_report(os.path.join(op_test_path, f"{k}.html")) for k,_ in VARIANTS}
        data_golden  = {k: parse_html_report(os.path.join(op_golden_path, f"{k}.html")) for k,_ in VARIANTS}

        # collect all model names
        models = set().union(*[d.keys() for d in data_test.values()], *[d.keys() for d in data_golden.values()])

        # write per-operator CSV
        out_path = os.path.join(out_dir, f"{op}.csv")
        header = ["Model name"]
        header += ["Status"]
        header += ["TIDL_Offload_test"]
        header += ["TIDL_Offload_golden"]
        reports_found = False

        for name, label in VARIANTS:
            report_test_path = os.path.join(op_test_path, f"{name}.html")
            report_golden_path = os.path.join(op_golden_path, f"{name}.html")
            if not (os.path.isfile(report_test_path) and os.path.isfile(report_golden_path)):
                continue

            reports_found = True
            col = label.replace(" ", "_")
            header += [f"{col}_test", f"{col}_golden"]
        
        if reports_found == False:
            print(f"Skipping {op} as reports are not present")
            continue

        op_cfg_dir = os.path.join(configs_dir, op)

        # 1. Read all CSVs under configs/<Operator>/
        model_attrs = {}   # model_name -> dict of attributes
        all_attr_keys = set()
        for fn in os.listdir(op_cfg_dir):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(op_cfg_dir, fn)
            with open(path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    mn = row["Model_Name"]
                    if mn not in model_attrs:
                        model_attrs[mn] = {}
                    # merge attributes (later CSVs can override)
                    for k,v in row.items():
                        if k == "Model_Name":
                            continue
                        model_attrs[mn][k] = v
                        all_attr_keys.add(k)

        print(op, all_attr_keys)
        header += sorted(all_attr_keys)

        num_upgraded = 0
        num_degraded = 0
        num_offload = 0
        total_tests = 0

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            # model-by-model rows
            for mn, attrs in sorted(model_attrs.items()):
                row = [mn]
                total_tests = total_tests +1
                check = 0
                for key,label in VARIANTS:
                    status_test, subgraphs, offload_test = data_test[key].get(mn, ("N/A", "N/A", "N/A"))
                    status_golden, subgraphs, offload_golden = data_golden[key].get(mn, ("N/A", "N/A", "N/A"))
                    test_found = True
                    golden_found = True
                    if status_test == "N/A":
                        test_found = False
                    if status_test == "N/A":
                        golden_found = False

                    if(status_test.lower()=="passed" and status_golden.lower()=="failed"):
                        check = 1
                    elif(status_test.lower()=="failed" and status_golden.lower()=="passed"):
                        check = -1
                if (test_found == False) or (golden_found == False):
                    row += ["N/A"]
                elif(check == 0):
                    row += ["Same"]
                elif(check==1):
                    row += ["Upgraded"]
                    num_upgraded = num_upgraded+1
                elif(check==(-1)):
                    row += ["Degraded"]
                    num_degraded = num_degraded+1
                
                row += [offload_test]
                row += [offload_golden]
                if(offload_test.lower() == "true"):
                    num_offload = num_offload+1
                for key, label in VARIANTS:
                    status_test, subgraphs, offload_test= data_test[key].get(mn, ("N/A", "N/A", "N/A"))
                    status_golden, subgraphs, offload_golden= data_golden[key].get(mn, ("N/A", "N/A", "N/A"))
                    row += [status_test, status_golden]

                for k in sorted(all_attr_keys):
                    row.append(attrs.get(k, ""))
                w.writerow(row)

        # record for the full summary
        op_summary = {"Operator": op, "Num Upgraded": num_upgraded, "Num Degraded": num_degraded, "TIDL_Offload_Percentage": num_offload, "Total_Tests": total_tests}
        total_val =0
        for key,_ in VARIANTS:
            da = data_test[key]
            if da:
                p = sum(1 for s,__,___ in da.values() if s.lower()=="passed")
                f = sum(1 for s,__,___ in da.values() if s.lower()!="passed")
                # op_summary[f"{key}_user_failures"] = f"{p}/{p+f}"
                op_summary[f"{key}_test_failures"] = f"{f}"
                total_val = p+f
            else:
                op_summary[f"{key}_test_failures"] = "-"
            dr = data_golden[key]
            if dr:
                p = sum(1 for s,__,___ in dr.values() if s.lower()=="passed")
                f = sum(1 for s,__,___ in dr.values() if s.lower()!="passed")
                op_summary[f"{key}_golden_failures"] = f"{f}"
                total_val = p+f
            else:
                op_summary[f"{key}_golden_failures"] = "-"
        op_summary["TIDL_Offload_Percentage"] = (num_offload/total_val)*100

        operator_summaries.append(op_summary)
        print(f"Generated {out_path}")

# --- FULL OPERATOR COMPARISON ---

VARIANTS = [
    ("compile_with_nc",        "Compile with NC"),
    ("compile_without_nc",     "Compile without NC"),
    ("infer_ref_with_nc",      "Infer Ref with NC"),
    ("infer_ref_without_nc",   "Infer Ref without NC"),
]

full_path = os.path.join(out_dir, "operator_test_report_summary.csv")
fields = ["Operator"] + ["Num Upgraded"] + ["Num Degraded"] + ["TIDL_Offload_Percentage"] + ["Total_Tests"] + ["compile_with_nc_test_failures"] + ["compile_with_nc_golden_failures"] + ["infer_ref_with_nc_test_failures"] + ["infer_ref_with_nc_golden_failures"] + ["compile_without_nc_test_failures"] + ["compile_without_nc_golden_failures"] + ["infer_ref_without_nc_test_failures"] + ["infer_ref_without_nc_golden_failures"]

if len(operator_summaries) > 0:
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in sorted(operator_summaries, key=lambda x: x["Operator"]):
            w.writerow(rec)

    print(f"Wrote consolidated comparison → {full_path}")
