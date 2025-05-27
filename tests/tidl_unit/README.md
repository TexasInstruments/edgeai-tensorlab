# TIDL Unit Tests
A production‑grade regression suite containing thousands of single‑operator ONNX models used to validate Texas Instruments Deep‑Learning (TIDL) kernels on any supported SoC.

## 1. Features
Fine‑grained coverage – Every operator / attribute / dtype combination is a separate minimal ONNX graph.<br>
Deterministic I/O – Golden inputs & outputs ship with each model, enabling bit‑exact comparison.<br>
Flexible execution – Run the full matrix or an ad‑hoc subset, locally or over NFS.<br>
CI‑ready – Generates CSV/HTML reports and supports pytest-xdist for parallel runs.

## 2. Prerequisites
| Dependency               | Minimum version | Notes |
|--------------------------|-----------------|-------|
| Python                   | 3.x             | Tested on 3.8 – 3.12 |
| pip                      | latest          | python -m pip install --upgrade pip |
| Python packages          | —               | Install once in a fresh pyenv/conda env: pip install -r requirements.txt |
| TIDL Models repo     	   | current `master`| Holds the ONNX operator assets |
| TIDL tools tar file      | —               | — |

**Setup on X86_PC**<br>
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
./setup_pc.sh  #inside main benchmark directory 
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

## 4. Obtaining tools
Generate/Fetch the tools tar ball for testing<br>
Update the tools path inside run_operator_test.sh<br>
```python
# Configuration
tools_path="<tidl_tools tarball path here>" #tools in name.tar.gz format
```
For taking tools from c7x use
```bash
tar -h -czvf tidl_tools.tar.gz tidl_tools/
# Now place this tools tar file path to the above tools_path 
```

## 5. Running the Tests

### 5.1 Full suite
```bash
./run_operator_test.sh <SOC>
```
&lt;SOC&gt; - AM62A, AM67A, AM68A, AM69A, TDA4VM 

### 5.2 Subset
Update the operators list inside run_operator_test.sh<br>
```python
# Configuration
OPERATORS=()
# Single operator like Max - OPERATORS=("Max")
# Multi operator like Softmax, Convolution & Sqrt - OPERATORS=("Softmax" "Convolution" "Sqrt")
# Full suite - OPERATORS=()
```

### 5.3 Additional Arguments
Operators and runtimes can be directly passed from command line using `--operators` and `--runtimes` respectively. Multiple space separated values can be passed.<br>
Accepted values for RUNTIMES are `onnxrt`, `tvmrt`. Default is `onnxrt`<br>
Operators accept either full operator suite names, or single layer names. (For eg. Both `MaxPool` and `Slice_1` are valid)
If not passed, OPERATORS and RUNTIMES list are taken from the script

```bash
# Examples

# Run tests for Add_1 and Add_2 layer in 'tvm' runtime
./run_operator_test.sh AM68A --runtimes tvmrt --operators Add_1 Add_2

# Run tests for Reshape model and Slice_3 layer in 'onnx' and 'tvm' runtimes
./run_operator_test.sh AM68A --runtimes tvmrt onnxrt --operators Slice_3 Reshape

# Run all tests in 'onnx' runtime. (Runtime defaults to 'onnx' if not passed)
./run_operator_test.sh AM68A
```


## 6. Repository Layout
```text
tidl_unit_tests/
├─ docs/                     	        # Usage notes
├─ logs/						        # pass/fail logs
├─ run_operator_test.sh                 # Operator testing script
├─ run_test.sh  				        # Main entry‑point script
├─ tidl_unit.yaml  				        # backend testing configuration
├─ tidl_unit_test_data/                 # Symlink → operator assets
├─ operator_test_report_comparison/     # CSV‑based comparison reports between runtimes
├─ operator_test_report_csv/            # CSV‑based intensive test reports
├─ operator_test_report_html/           # HTML reports
├─ report_script/                       # Report‑generation scripts'
├─ requirements.txt  			        # python requirements
... other pytest requirements
```

## 7. Reports Layout
```text
tidl_unit_tests/
├── operator_test_report_comparison/                # Comparison between runtimes
│   ├── onnxrt/
│   │   ├── <Operator_Name>/
│   │   └── …
│   ├── tvmrt/
│   │   └── …
│   ├── compile_with_nc_comparison.csv              # Compilation comparion (with NC)
│   ├── compile_without_nc_comparison.csv           # Compilation comparion (without NC)
│   ├── infer_with_nc_comparison.csv                # Inference comparion (with NC)
│   ├── infer_without_nc_comparison.csv             # Inference comparion (without NC)
│   ├── with_nc_comparison.csv                      # Combined comparison (with NC)
│   └── without_nc_comparison.csv                   # Complete comparion (without NC)
├── operator_test_report_csv/
│   ├── onnxrt/                                     # Reports for ONNX Runtime
│   │   ├── complete_test_reports/                  # Customer‑facing reports 
│   │   │   ├── <Operator_Name>.csv                 # Operator‑specific report
│   │   │   └── operator_test_report_summary.csv    # Aggregate summary
│   │   └── customer_test_reports/                  # Customer‑facing reports 
│   │       ├── …
│   │       └── …
│   └── tvmrt/                                      # Reports for TVM Runtime
│       ├── …
│       └── …
└── operator_test_report_html/               
    ├── onnxrt/
    │   ├── <Operator_Name>/
    │   └── …
    └── tvmrt/
        └── …
```

## 8. Comparison Script
Generate performance comparison between runtimes using `run_operator_comparison.py`


### 8.1 Running the script
```bash
python3 run_operator_comparison.py
```
This generates test reports under `./operator_test_report_comparison/`<br>
```text
Optional Arguments:
    --runtime <RUNTIMES>    : Runtimes to run. If left empty, runs all runtimes defined under ALL_RUNTIMES inside the script
    --operator <OPERATORS>  : Operators to run. If left empty, runs full suite.
    --compare               : Compare mode. Doesnt execute tests, uses existing reports to generate comparison report.    
```


### 8.2 Customisation
- Set device by setting `DEVICE` inside the script. Accepts allowed devices mentioned in section 5.1<br>
- Change report location by changing `REPORT_DIR` and `OUT_DIR`<br>
- Add runtimes by adding to `ALL_RUNTIMES` and `reports` dictionary.
- Add known error patterns to `error_regex` list to capture them in the report.

```bash
# Examples

# Runs the tvmrt runtine tests for Relu and Max
python3 run_operator_comparison.py --runtime tvmrt --operator Relu Max
    
# Runs tests for Add for all the runtimes
python3 run_operator_comparison.py --operator Add
    
# Runs comparison for Convolution
python3 run_operator_comparison.py --compare --operator Convolution
```

## 9. Documentation

Usage notes: [usage-notes.md](docs/usage-notes.md)<br>
Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)<br>
Code Outline: [code-outline.md](docs/code-outline.md)


