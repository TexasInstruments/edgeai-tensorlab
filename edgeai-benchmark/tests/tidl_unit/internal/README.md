# TIDL backend test framework
A unified pytest based test environment to test and verify Texas Instruments Deep-Learning (TIDL) models and library.

## 1. Features
- Evaluate compilation and inference of your own models to check readiness for deployement to TI SOC <br>
- Generates HTML reports and supports pytest-xdist for parallel runs.

## 2. Prerequisites
| Dependency               | Minimum version | Notes |
|--------------------------|-----------------|-------|
| Python                   | 3.x             | Tested on 3.10(x86) and 3.12(EVM) |
| pip                      | latest          | python -m pip install --upgrade pip |
| Python packages          | —               | Install once in a fresh pyenv/conda env: pip install -r requirements.txt |

## 3. Setup on X86_PC

Note: Recommended to run this in a fresh python virtual environment

Install pyenv:<br>
```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc
exec ${SHELL}
```
Create and activate pyenv:<br>
```bash
pyenv install 3.10
pyenv virtualenv 3.10 benchmark
pyenv activate benchmark
pip install --upgrade pip setuptools
```

Run Setup:<br>
```bash
./setup_pc.sh  #inside main benchmark directory 
cd tests/tidl_unit/
pip install -r requirements.txt
```

## Operator Tests
### Operator & Config Assets
```bash
ln -s <path/to/tidl_models>/unitTest/onnx/tidl_unit_test_assets/operators <path/to/edgeai-benchmark>tests/tidl_unit/tidl_unit_test_data/operators
ln -s <path/to/tidl_models>/unitTest/onnx/tidl_unit_test_assets/configs <path/to/edgeai-benchmark>tests/tidl_unit/tidl_unit_test_data/configs
```

### Running Operator Tests
```bash
./run_model_test.sh
```
This generates test reports under `operator_test_reports/<runtime>/<SOC>/`

```bash
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
    --nmse_threshold            Normalized Mean Squared Error (NMSE) thresehold for inference output. Default: 0.5
    --operators                 List of operators (space separated string) to run. By default every operator under tidl_unit_test_data/operators
    --runtimes                  List of runtimes (space separated string) to run tests. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tidl_tools_path           Path of tidl tools tarball (named as tidl_tools.tar.gz)
    --compiled_artifacts_path   Path of compiled model artifacts. Will be used only for TARGET run.

    Example:
        ./run_operator_test.sh --SOC=AM68A --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --operators=\"Add Mul Sqrt\" --runtimes=\"onnxrt\"
        This will run unit tests for (Add, Mul, Sqrt) operators on AM68A using onnxrt runtime, aritifacts will be saved and will run Host emulation inference 
    "
```

### Operator Test Usage Examples

```bash
# Run all operators under <OPERATOR_GROUP_1> AND <OPERATOR_GROUP_2> for specific models on AM68A using onnxrt runtime with compilation (with nc only) and host emulation ref inference
./run_operator_test.sh --SOC=AM68A --compile_without_nc=0 --compile_with_nc=1 --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --models="<OPERATOR_GROUP_1> <OPERATOR_GROUP_2>"

# Run tests specified in a test file on AM68A
./run_operator_test.sh --SOC=AM68A --compile_without_nc=0 --compile_with_nc=1  --run_ref=1 --run_natc=0 --run_ci=0 --test_file=operator_to_test.txt --runtimes="onnxrt"

# Run all models under tidl_unit_test_data/operators folder
./run_operator_test.sh --SOC=AM68A --compile_without_nc=0 --compile_with_nc=1  --run_ref=1 --run_natc=0 --run_ci=0 --runtimes="onnxrt"
```

#### Operator Test File Format
When using the `--test_file` option, the file should contain one model per line in the format `<specific_model>` and end with a blank new line, for example:
```
# Lines beginning with # are treated as comments
Convolution_1
MaxPool_5

```

### Operator Comparison Script
Generate performance comparison between runtimes using `run_operator_comparison.py`

#### Running the script
```bash
python3 run_operator_comparison.py
```
This generates test reports under `../operator_test_report/comparison/`<br>
```text
Optional Arguments:
    --runtime <RUNTIMES>    : Runtimes to run. If left empty, runs all runtimes defined under ALL_RUNTIMES inside the script
    --operator <OPERATORS>  : Operators to run. If left empty, runs full suite.
    --compare               : Compare mode. Doesnt execute tests, uses existing reports to generate comparison report.   

If single tests are to be run, then only that should be passed.
```

#### Customisation
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
    
# Runs comparison for Conv
python3 run_operator_comparison.py --compare --operator Conv

#Runs onnx and tvm tests for just MaxPool_2 test 
python3 run_operator_comparison.py --operator MaxPool_2
```

## Model Tests
### Model Assets
```bash
ln -s <path/to/tidl_models>/unitTest/onnx/tidl_unit_test_assets/models <path/to/edgeai-benchmark>/tests/tidl_unit/tidl_unit_test_data/models
```

### Running Model Tests
```bash
./run_model_test.sh
```
This generates test reports under `model_test_reports/<runtime>/<SOC>/`

```text
Usage:
    Helper script to run unit tests model wise

    Options:
    --SOC                       SOC. Allowed values are (AM62A, AM67A, AM68A, AM69A, TDA4VM)
    --tidl_offload              Offload tests to TIDL. Allowed values are (0,1). Default=1
    --compile_without_nc        Compile models without NC. Allowed values are (0,1). Default=0
    --compile_with_nc           Compile models with NC. Allowed values are (0,1). Default=1
    --run_ref                   Run HOST emulation inference. Allowed values are (0,1). Default=1
    --run_natc                  Run Inference with NATC flow control. Allowed values are (0,1). Default=0
    --run_ci                    Run Inference with CI flow control. Allowed values are (0,1). Default=0
    --work_dir                  Full path to save model artifacts during compilation. Same will be used to fetch compiled model artifacts during inference
                                Default is work_dirs/modelartifacts
    --save_model_artifacts      Whether to preserve compiled artifacts or not in work_dir. Allowed values are (0,1). Default=0
    --temp_buffer_dir           Path to redirect temporary buffers for x86 runs. Default is /dev/shm
    --temp_nc_dir               Path to redirect temporary NC buffers. Default is /tmp
    --nmse_threshold            Normalized Mean Squared Error (NMSE) threshold for inference output. Default: Parsed from config.yaml of model is present else 0.5
    --runtimes                  List of runtimes (space separated string) to run tests. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tensor_bits               8/16/32. Default: 8
    --num_threads               Number of threads for test run. Default=auto
    --tidl_tools_path           Path of tidl tools tarball
    --test_file                 Path to text file containg test to run. Default=null
    --models                    List of models (space separated string) to run. By default every model under tidl_unit_test_data/models
```

### Model Test Usage Examples

```bash
# Run all models under <MODEL_GROUP_1> AND <MODEL_GROUP_2> for specific models on AM68A using onnxrt runtime with compilation (with nc only) and host emulation ref inference
./run_model_test.sh --SOC=AM68A --compile_without_nc=0 --compile_with_nc=1 --run_ref=1 --run_natc=0 --run_ci=0 --save_model_artifacts=1 --models="<MODEL_GROUP_1> <MODEL_GROUP_2>"

# Run tests specified in a test file on AM68A
./run_model_test.sh --SOC=AM68A --compile_without_nc=0 --compile_with_nc=1  --run_ref=1 --run_natc=0 --run_ci=0 --test_file=models_to_test.txt --runtimes="onnxrt"

# Run all models under tidl_unit_test_data/models folder
./run_model_test.sh --SOC=AM68A --compile_without_nc=0 --compile_with_nc=1  --run_ref=1 --run_natc=0 --run_ci=0 --runtimes="onnxrt"
```

#### Model Test File Format
When using the `--test_file` option, the file should contain one model per line in the format `<parent_model_folder>/<specific_model>` and end with a blank new line, for example:
```
# Lines beginning with # are treated as comments
MobileNet/mobilenet_v1
ResNet50/resnet50_v1.5

```

## Summary Report Generation
A summary report is also automatically generated at the end of both operator and model test using the `report_summary_generation.py` script. This script provide an overview of all test results.

The script is generated at `model_test_reports/<runtime>/<SOC>/` for model tests and `operator_test_reports/<runtime>/<SOC>/` for operator tests

To use this script manually 
```python
python3 report_summary_generation.py --reports_path=operator_test_reports/<runtime>/ # For operator test

python3 report_summary_generation.py --reports_path=model_test_reports/<runtime>/ # For model test
```

The script will generate summary report for all `<SOC>` under the <reports>/<runtime> folder