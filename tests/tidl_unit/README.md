# TIDL Unit Tests
A production‑grade regression suite containing thousands of single‑operator ONNX models used to validate Texas Instruments Deep‑Learning (TIDL) kernels on any supported SoC.

## 1. Features
Fine‑grained coverage – Every operator / attribute / dtype combination is a separate minimal ONNX graph.
Deterministic I/O – Golden inputs & outputs ship with each model, enabling bit‑exact comparison.
Flexible execution – Run the full matrix or an ad‑hoc subset, locally or over NFS.
CI‑ready – Generates JUnit/HTML reports and supports pytest-xdist for parallel runs.

## 2. Prerequisites
| Dependency               | Minimum version | Notes |
|------------              |-----------------|-------|
| Python                   | 3.x             | Tested on 3.8 – 3.12 |
| pip                      | latest          | python -m pip install --upgrade pip |
| Python packages          | —               | Install once in a fresh venv/conda env: pip install -r requirements.txt \ pytest pytest-xdist pytest-html==3.2.0 |
| **TIDL Models repo**     | current `main`  | Holds the ONNX operator assets |

---

## 3. Obtaining Operator Assets
<!-- ```bash -->
# Clone (anywhere)
git clone <tidl_models_repo>
export TIDL_OPS=$PWD/tidl_models/unitTest/onnx/tidl_unit_test_assets/operators

### 3.1 Local symbolic‑link (dev workflow)
rm -rf tidl_unit_test_data/operator          # purge any stale link/dir
ln -s "${TIDL_OPS}" tidl_unit_test_data/operator

### 3.2 NFS mount (CI / farm)

## 4. Running the Tests

### 4.1 Full suite
./run_operator_test.sh <SOC>
<SOC> - AM62A, AM67A, AM68A, AM69A, TDA4VM 

### 4.2 Subset
Edit the OPERATORS=( … ) array inside run_operator_test.sh:

Single operator like Max - OPERATORS=("Max")
Multi operator like Softmax, Convolution & Sqrt - OPERATORS=("Softmax" "Convolution" "Sqrt")
full suite - OPERATORS=()

## 5. Repository Layout
tidl_unit_tests/

├─ docs/                        # Usage notes, pass/fail logs, code outline
├─ run_operator_test.sh         # Main entry‑point script
├─ tidl_unit_test_data/         # Symlink → operator assets
├─ tests/                       # PyTest collections & fixtures
├─ operator_test_report_csv/    # Csv based intensive test reports    
├─ operator_test_report_html/   # Html based onnx backed default reports
├─ report_script/               # Report generation scripts

## 6. Reports Layout
tidl_unit_tests/

├─ operator_test_report_csv/    
    ├─ customer_test_reports/                   # Customore Facing Reports
        ├─ <Operator_Name>.csv                  # Operator Specific Report
        └─ operator_test_report_summary.csv     # Summary Report
    ├─ comaparison_test_reports/                # Referance comparison Report
        ├─ ...
        └─ ...
    ├─ absolute_test_repors/                    # Operator Test Report
        ├─ ...
        └─ ...

## 7. Documentation

Usage notes: [usage-notes.md](docs/usage-notes.md)
Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)
Code Outline: [code-outline.md](docs/code-outline.md)


