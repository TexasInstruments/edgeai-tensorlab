# TIDL backend test platform
A unified environment that accepts models in the required format to verify Texas Instruments Deep-Learning (TIDL) kernels on any supported SoC.

## 1. Features
Deterministic I/O – Golden inputs & outputs ship with each model, enabling bit‑exact comparison.<br>
Flexible execution – Run the full matrix or an ad‑hoc subset, locally or over NFS.<br>
CI‑ready – Generates HTML reports and supports pytest-xdist for parallel runs.

## 2. Prerequisites
| Dependency               | Minimum version | Notes |
|--------------------------|-----------------|-------|
| Python                   | 3.x             | Tested on 3.8 – 3.12 |
| pip                      | latest          | python -m pip install --upgrade pip |
| Python packages          | —               | Install once in a fresh pyenv/conda env: pip install -r requirements.txt |

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

## 3. Structuring models and input/output buffer diectory 
```
inside tidl_unit_test_data/
├── operator/         ← Generated ONNX models & I/O protobufs
│   └── model_testing_subset_directory_1/
│       ├── model_1/
│       │   ├── model.onnx
│       │   ├── test_data_set_0/
│       │   │   ├── input_0.pb
│       │   │   ├──	input_1.pb ...  # incase of multiple inputs
│       │   │   ├── output_0.pb
│       └── model_2/
│           ├── ...
│   └── model_testing_subset_directory_2/
│       ├── model_3/...
```

## 4. Running the Tests
```bash
cd ../../ #inside the main benchmark directory
source ./run_set_env.sh <SOC>
# Excepted SOC's - AM62A, AM67A, AM68A, AM69A, TDA4VM
cd tests/tidl_unit
./run_test.sh 
"Usage:
    Helper script to invoke pytest for tidl unit tests

    To change the custom configuration, modify ./scripts/benchmark_custom.py.

    Options:
    --test_suite        Test suite. Allowed values are (operator) i.e name of the parent directory of models inside tidl_unit_test_data/
    --run_compile       Run model compilation test. Allowed values are (0,1). Default=1.
    --run_infer         Run model inference test. Allowed values are (0,1). Default=1.
    --tidl_offload      Enable TIDL Offload. Allowed values are (0,1). Default=1.
    --runtime           Select the Compiler Runtime to use. Allowed values are (onnxrt, tvmrt). Default=onnxrt
    --tests             Specify tests name. If null, will run all test based on test_suite. Default=null.
                        TEST_SUITE:
                            operator: You can specify comma seperated operator name (Ex: model_testing_subset_directory_1) or specific test (Ex: model_testing_subset_directory_2) 

    Example:
    TEST_SUITE:
        operator: ./run_test.sh --test_suite=operator --tests=model_testing_subset_directory_1,model_3,model_testing_subset_directory_3 --run_compile=1 --run_infer=1
                   This will run all tests under model_testing_subset_directory_1 and model_testing_subset_directory_3 and also model_3 test individually.
    "
```

## 5. Repository Layout
```text
tidl_unit/
├─ docs/                     	        # Usage notes
├─ logs/						        # pass/fail logs
├─ run_test.sh  				        # Main entry‑point script
├─ tidl_unit.yaml  				        # backend testing configuration
├─ tidl_unit_test_data/                 # Symlink → operator assets
├─ work_dirs/                 			# Artifacts directory
├─ requirements.txt  			        # python requirements
├─ internal/  			        		# internal use case
... other pytest requirements
```

## 6. Documentation

Usage notes: [usage-notes.md](docs/usage-notes.md)<br>
Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)<br>
Code Outline: [code-outline.md](docs/code-outline.md)
