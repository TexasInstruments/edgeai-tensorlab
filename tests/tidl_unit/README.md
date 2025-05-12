# TIDL Unit Tests
A production‑grade regression suite containing thousands of single‑operator ONNX models used to validate Texas Instruments Deep‑Learning (TIDL) kernels on any supported SoC.

## 1. Features
Fine‑grained coverage – Every operator / attribute / dtype combination is a separate minimal ONNX graph.<br>
Deterministic I/O – Golden inputs & outputs ship with each model, enabling bit‑exact comparison.<br>
Flexible execution – Run the full matrix or an ad‑hoc subset, locally or over NFS.<br>
CI‑ready – Generates JUnit/HTML reports and supports pytest-xdist for parallel runs.

## 2. Prerequisites
| Dependency               | Minimum version | Notes |
|------------              |-----------------|-------|
| Python                   | 3.x             | Tested on 3.8 – 3.12 |
| pip                      | latest          | python -m pip install --upgrade pip |
| Python packages          | —               | Install once in a fresh pyenv/conda env: pip install -r requirements.txt |
| **TIDL Models repo**     | current `main`  | Holds the ONNX operator assets |

**Setup on X86_PC**C<br>
Install pyenv using the following command.<br>
```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc
exec ${SHELL}
```
Create and activate pyenv.<br>
```bash
pyenv install 3.10
pyenv virtualenv 3.10 benchmark
pyenv activate benchmark
pip install --upgrade pip setuptools
```
Setup scripts.<br>
```bash
./setup_pc.sh
cd tests/tidl_unit/
pip install -r requirements.txt
```

## 3. Obtaining Operator Assets
```bash
Clone (anywhere)
git clone <tidl_models_repo>
export TIDL_OPS=$PWD/tidl_models/unitTest/onnx/tidl_unit_test_assets/operators
```

### 3.1 Local symbolic‑link (dev workflow)
```bash
rm -rf tidl_unit_test_data/operator          # purge any stale link/dir
ln -s "${TIDL_OPS}" tidl_unit_test_data/operator
```

### 3.2 NFS mount (CI / farm)
Mount from a local device with nfs mount

## 4. Running the Tests

### 4.1 Full suite
```bash
./run_operator_test.sh <SOC>
```
&lt;SOC&gt; - AM62A, AM67A, AM68A, AM69A, TDA4VM 

### 4.2 Subset
Edit the OPERATORS=( … ) array inside run_operator_test.sh:<br>
Single operator like Max - OPERATORS=("Max")<br>
Multi operator like Softmax, Convolution & Sqrt - OPERATORS=("Softmax" "Convolution" "Sqrt")<br>
full suite - OPERATORS=()

## 5. Repository Layout
```text
tidl_unit_tests/
├─ docs/                     	# Usage notes
├─ logs/						# pass/fail logs
├─ run_operator_test.sh         # Operator testing script
├─ run_test.sh  				# Main entry‑point script
├─ tidl_unit.yaml  				# backend testing configuration
├─ tidl_unit_test_data/         # Symlink → operator assets
├─ operator_test_report_csv/    # CSV‑based intensive test reports
├─ operator_test_report_html/   # HTML reports (ONNX‑backed, default)
├─ report_script/               # Report‑generation scripts'
├─ requirements.txt  			# python requirements
... other pytest requirements
```

## 6. Reports Layout
```text
tidl_unit_tests/
├── operator_test_report_csv/
│   ├── customer_test_reports/                # Customer‑facing reports
│   │   ├── <Operator_Name>.csv               # Operator‑specific report
│   │   └── operator_test_report_summary.csv  # Aggregate summary
│   ├── comparison_test_reports/              # Reference comparison reports
│   │   ├── …
│   │   └── …
│   └── absolute_test_reports/                # Full operator test reports
│       ├── …
│       └── …
└── operator_test_report_html/               
```

## 7. Documentation

Usage notes: [usage-notes.md](docs/usage-notes.md)<br>
Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)<br>
Code Outline: [code-outline.md](docs/code-outline.md)


