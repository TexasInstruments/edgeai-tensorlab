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

## 3. Obtaining Operator & Config Assets
```bash
Clone (anywhere)
git clone <tidl_models_repo>
export TIDL_OPS=$PWD/tidl_models/unitTest/onnx/tidl_unit_test_assets/operators
export TIDL_CONFIGS=$PWD/tidl_models/unitTest/onnx/tidl_unit_test_assets/configs
```

### 3.1 Local symbolic‑link (dev workflow)
```bash
ln -s "${TIDL_OPS}" tidl_unit_test_data/operator
ln -s "${TIDL_CONFIGS}" tidl_unit_test_data/configs
```

### 3.2 NFS mount (CI / farm)
Mount from a local device with nfs mount

## 4. Obtaining tools
Generate/Fetch the tools tar ball for testing<br>
Update the tools path inside run_operator_test.sh<br>
```python
# Configuration
tools_path="<tidl_tools tarball path here>" # tools in tidl_tools.tar.gz format
```
For taking tools from c7x use
```bash
tar -h -czvf tidl_tools.tar.gz tidl_tools/
# Now place this tools tar file path to the above tools_path 
```

## 5. Running the Tests

```bash
cd internal/
./run_operator_test.sh 
"Usage:
    Helper script to run unit tests operator wise

    Options:
    --SOC                       SOC. Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)
    --compile_without_nc        Compile models without NC. Allowed values are (0,1). Default=0
    --compile_with_nc           Compile models with NC. Allowed values are (0,1). Default=1
    --run_ref                   Run HOST emulation inference. Allowed values are (0,1). Default=1
    --run_natc                  Run Inference with NATC flow control. Allowed values are (0,1). Default=0
    --run_ci                    Run Inference with CI flow control. Allowed values are (0,1). Default=0
    --run_target                Run Inference on TARGET. Allowed values are (0,1). Default=0
    --save_model_artifacts      Whether to save compiled artifacts or not. Allowed values are (0,1). Default=0
    --save_model_artifacts_dir  Path to save model artifacts if save_model_artifacts is 1. Default is work_dirs/modelartifacts
    --temp_buffer_dir           Path to redirect temporary buffers for x86 runs. Default is /dev/shm
    --operators                 List of operators (space separated string) to run. By default every operator under tidl_unit_test_data/operators
    --runtimes                  List of runtimes (space separated string) to run tests. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tidl_tools_path           Path of tidl tools tarball (named as tidl_tools.tar.gz)
    --compiled_artifacts_path   Path of compiled model artifacts. Will be used only for TARGET run.

    Example:
        ./run_operator_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --operators=\"Add Mul Sqrt\" --runtimes=\"onnxrt\"
        This will run unit tests for (Add, Mul, Sqrt) operators on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 
    "
cd ..
```

## 6. Repository Layout
```text
tidl_unit/internal/
├─ run_operator_test.sh                 # Operator testing script
├─ operator_test_report_comparison/     # CSV‑based comparison reports between runtimes
├─ operator_test_report_csv/            # CSV‑based intensive test reports
├─ operator_test_report_html/           # HTML reports
├─ report_script/                       # Report‑generation scripts
... other pytest requirements
```

## 7. Reports Layout
```text
tidl_unit_tests/
├── operator_test_report_csv/
│   ├── comparison/                                 # Reports for ONNX Runtime
│   │   ├── compile_with_nc_comparison.csv          # Compilation comparion (with NC)
│   │   ├── compile_without_nc_comparison.csv       # Compilation comparion (without NC)
│   │   ├── infer_with_nc_comparison.csv            # Inference comparion (with NC)
│   │   ├── infer_without_nc_comparison.csv         # Inference comparion (without NC)
│   │   ├── with_nc_comparison.csv                  # Combined comparison (with NC)
│   │   └── without_nc_comparison.csv               # Complete comparion (without NC)
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
This generates test reports under `../operator_test_report_csv/comparison/`<br>
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

