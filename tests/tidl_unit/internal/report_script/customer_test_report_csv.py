#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys

def parse_html_report(html_path):
    """
    Returns a dict mapping model_name -> (status, reason, subgraphs, offload)
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
        # extract the bracketed model name
        if "[" in test_cell and "]" in test_cell:
            mn = test_cell.split("[",1)[1].split("]",1)[0]
        else:
            mn = test_cell

        reason = row.select_one("div.log").get_text(strip=True)

        results[mn] = (status, reason, subgraphs, offload)
    return results


# --- CONFIGURATION ---

A_DIR   = "operator_test_reports"
R_DIR   = "operator_test_report_ref"
CONFIGS_DIR = "../../tidl_unit_test_data/configs"
OUT_DIR = "customer_test_reports"
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = [
    ("infer_with_nc",      "Inferance Test"),
]

operator_summaries = []

for op in sorted(os.listdir(A_DIR)):
    dir_a = os.path.join(A_DIR, op)

    mod_num = None
    if '_' in op:
        mod_num = op.split('_')[-1]
        op = "_".join(op.split('_')[:-1])

    data_a = {k: parse_html_report(os.path.join(dir_a, f"{k}.html")) for k,_ in VARIANTS}

    # collect all model names from parsed HTML reports
    models = set().union(*[d.keys() for d in data_a.values()], *[])

    out_path = os.path.join(OUT_DIR, f"{op}.csv")
    header = ["model name"]
    header += ["TIDL_Offload"]
    for _, label in VARIANTS:
        col = label.replace(" ", "_")
        header += [f"{col}"]

    op_cfg_dir = os.path.join(CONFIGS_DIR, op)

    model_attrs = {}
    all_attr_keys = set()

    # Only read configs if directory exists
    if os.path.isdir(op_cfg_dir):
        for fn in os.listdir(op_cfg_dir):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(op_cfg_dir, fn)
            with open(path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    mn = row["model name"]
                    if mod_num and f"{op}_{mod_num}" != mn:
                        continue
                    if mn not in model_attrs:
                        model_attrs[mn] = {}
                    for k,v in row.items():
                        if k == "model name":
                            continue
                        model_attrs[mn][k] = v
                        all_attr_keys.add(k)

        header += sorted(all_attr_keys)
    else:
        # config dir missing - skip config columns
        model_attrs = {}
        all_attr_keys = set()

    num_offload = 0
    total_tests = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        # If configs present, use models from model_attrs, else from parsed data
        if model_attrs:
            model_list = sorted(model_attrs.keys())
        else:
            model_list = sorted(models)

        for mn in model_list:
            attrs = model_attrs.get(mn, {})
            total_tests += 1
            row = [mn]

            # Find offload status from any variant available
            offload_val = "-"
            for key, _ in VARIANTS:
                if mn in data_a[key]:
                    offload_val = data_a[key][mn][3].lower()
                    break

            if offload_val == "true":
                num_offload += 1
                row.append("true")
            else:
                row.append("false")

            for key, _ in VARIANTS:
                sa, ra, subg, offload = data_a[key].get(mn, ("-", "", "", ""))
                row.append(sa)

            for k in sorted(all_attr_keys):
                row.append(attrs.get(k, ""))

            w.writerow(row)

    op_summary = {"Operator": op, "TIDL_Offload_Percentage": num_offload, "Total_Tests": total_tests}
    total_val = 0
    for key, _ in VARIANTS:
        da = data_a[key]
        if da:
            p = sum(1 for s,_,__,___ in da.values() if s.lower() == "passed")
            f = sum(1 for s,_,__,___ in da.values() if s.lower() != "passed")
            op_summary["Inferance_test_result_passed"] = str(p)
            total_val = p + f
        else:
            op_summary["Inferance_test_result_passed"] = "-"

    # Avoid division by zero
    if total_val > 0:
        op_summary["TIDL_Offload_Percentage"] = (num_offload / total_val) * 100
    else:
        op_summary["TIDL_Offload_Percentage"] = 0

    operator_summaries.append(op_summary)
    print(f"Wrote {out_path}")

# --- FULL OPERATOR COMPARISON ---

full_path = os.path.join(OUT_DIR, "operator_test_report_summary.csv")
fields = ["Operator", "TIDL_Offload_Percentage", "Total_Tests", "Inferance_test_result_passed"]

with open(full_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in sorted(operator_summaries, key=lambda x: x["Operator"]):
        w.writerow(rec)

print(f"Wrote consolidated comparison → {full_path}")