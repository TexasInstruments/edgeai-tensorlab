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

## 4. Operator & Config Assets
```bash
ln -s <path/to/tidl_models>/unitTest/onnx/tidl_unit_test_assets/operators <path/to/edgeai-benchmark>tests/tidl_unit/tidl_unit_test_data/operators
ln -s <path/to/tidl_models>/unitTest/onnx/tidl_unit_test_assets/configs <path/to/edgeai-benchmark>tests/tidl_unit/tidl_unit_test_data/configs
```

## 5. Running the Tests

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
cd ..
```

## 6. Comparison Script
Generate performance comparison between runtimes using `run_operator_comparison.py`

### 6.1 Running the script
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

### 6.2 Customisation
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
