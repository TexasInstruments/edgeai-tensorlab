#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys

def parse_html_report(html_path):
    """
    Returns a dict mapping model_name -> (status, reason)
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

        # get the last non-empty line of the log
        reason = row.select_one("div.log").get_text(strip=True)

        results[mn] = (status, reason, subgraphs, offload)
    return results


# --- CONFIGURATION ---

A_DIR   = "operator_test_reports"
R_DIR   = "operator_test_report_ref"
CONFIGS_DIR = "configs"
OUT_DIR = "complete_test_reports"
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = [
    ("compile_with_nc",    "Compile with NC"),
    ("compile_without_nc", "Compile without NC"),
    ("infer_with_nc",      "Infer with NC"),
    ("infer_without_nc",   "Infer without NC"),
]

# Will accumulate per-operator summary for the final CSV
operator_summaries = []

for op in sorted(os.listdir(A_DIR)):
    dir_a = os.path.join(A_DIR,   op)

    # parse both sets
    data_a = {k: parse_html_report(os.path.join(dir_a, f"{k}.html")) for k,_ in VARIANTS}

    # collect all model names
    models = set().union(*[d.keys() for d in data_a.values()], *[])

    # write per-operator CSV
    out_path = os.path.join(OUT_DIR, f"{op}.csv")
    header = ["Model_name"]
    header += ["TIDL_Offload"]
    for _, label in VARIANTS:
        col = label.replace(" ", "_")
        # header += [f"{col}_user", f"{col}_ref"]
        header += [f"{col}", f"{col}_logs"]

    op_cfg_dir = os.path.join(CONFIGS_DIR, op)

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

    header += sorted(all_attr_keys)

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
                sa, ra, subg, offload = data_a[key].get(mn, ("-", "", "", ""))
            row += [offload.lower()]
            if(offload.lower()=="true"):
                num_offload = num_offload+1
            for key,label in VARIANTS:
                sa, ra, subg, offload= data_a[key].get(mn, ("-", "", "", ""))
                row += [sa, ra]
            for k in sorted(all_attr_keys):
                row.append(attrs.get(k, ""))
            w.writerow(row)

    # record for the full summary
    op_summary = {"Operator": op, "TIDL_Offload_Percentage": num_offload, "Total_Tests": total_tests}
    total_val =0
    for key,_ in VARIANTS:
        da = data_a[key]
        if da:
            p = sum(1 for s,_,__,___ in da.values() if s.lower()=="passed")
            f = sum(1 for s,_,__,___ in da.values() if s.lower()!="passed")
            op_summary[f"{key}_failures"] = f"{f}"
            total_val = p+f
        else:
            op_summary[f"{key}_failures"] = "-"
    op_summary["TIDL_Offload_Percentage"] = (num_offload/total_val)*100

    operator_summaries.append(op_summary)
    print(f"Wrote {out_path}")

# --- FULL OPERATOR COMPARISON ---

VARIANTS = [
    ("compile_with_nc",    "Compile with NC"),
    ("compile_without_nc", "Compile without NC"),
    ("infer_with_nc",      "Infer with NC"),
    ("infer_without_nc",   "Infer without NC"),
]

full_path = os.path.join(OUT_DIR, "operator_test_report_summary.csv")
fields = ["Operator"] + ["TIDL_Offload_Percentage"] + ["Total_Tests"] + ["compile_with_nc_failures"] + ["infer_with_nc_failures"] + ["compile_without_nc_failures"] + ["infer_without_nc_failures"] 

with open(full_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in sorted(operator_summaries, key=lambda x: x["Operator"]):
        w.writerow(rec)

print(f"Wrote consolidated comparison → {full_path}")