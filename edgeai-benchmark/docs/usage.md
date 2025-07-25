## Introduction
Compiling a DNN model is the process of quantizing and converting the model into a format that can be inferred with TIDL. TIDL (and its open source front ends) provides utilities to compile models. The imported artifacts can then be used to run inference.

In addition to what is provided with TIDL, this repository provides igh level utilities for compiling DNN models. These tools include dataset loaders, pre-processing utilities, post-processing utilities and metric computation utilities.


## Usage
[run_benchmarks_pc.sh](../run_benchmarks_pc.sh) is the main script in this repository that does compilation of models and benchmark. Usage is:
```
run_benchmarks_pc.sh <SOC>
```

For example, to run compilation for AM68A, this would be the command:
```
run_benchmarks_pc.sh AM68A
```

It uses parallel processing to compile multiple models in parallel and is especially useful while compiling several models, such as all the models in the Model Zoo. The number of parallel processes used defaults to 16 and is set in [settings_import_on_pc.yaml](../settings_import_on_pc.yaml). Change it to a different value if needed. It can be done by either modifying this settings file or by using the --parallel_processes commandline argument.

**Model compilation can be run only on PC. The EVM/device does not support model compilation. However, the inference of a compiled model can be run on PC or on device.**


## Compiling models in the Model Zoo
* **model_shortlist** in the [settings_base.yaml](../settings_base.yaml) file indicates which all models are run by default. Many models have a shortlist or a priority assigned. If model_shortlist parameter is set to 100, then only those models with model_shortlist priority value less thatn or equal to 100 will be run. This values can be changed by either changing it directly in the settings file or by passing as an argument to run_benchmarks_pc.sh
* **modelartifacts_path** in the [settings_base.yaml](../settings_base.yaml) file indicates the location where the artifacts are generated or expected. It currently points to work_dirs/modelartifacts/<SOC>/
* Each model needs a set of side information for compilation. The [configs module](../configs) in this repository by default to understand this information. 
* But this script can also use a [configs.yaml](https://github.com/TexasInstruments/edgeai-tensorlab/blob/main/edgeai-modelzoo/models/configs.yaml) file (instead of the configs module) by specifying it in the argument --configs_path.


## Compiiling models with Firmware update features enabled
Versions of tidl_tools released after an SDK release may come with improvements that may not be compatible - so those improvements may not be enabled by default. In order to compile models with those improvements, firmware version can be provided. Run using:
```
run_benchmarks_pc.sh <SOC> --c7x_firmware_version <firmwareversion>
```

For example, compiling for latest firmware version in the 10.1 release series could be done by:
```
run_benchmarks_pc.sh AM68A --c7x_firmware_version 10_01_04_00
```

But such models may require the SDK firmware to be updated - otherwise those compiled models may not run on EVM. See more details of compatibility in: 
- [version compatibility table](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md)
- [how to update the firmware](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/update_target.md)


## Running inference / benchmark on *PC* using pre-compiled model artifacts
* For the models in the Model Zoo, links to pre-compiled artifacts may be provided in edgeai-modelzoo/modelartifacts. If you would like to use these pre-compiled compiled artifacts and only do inference, then create a symbolic link called modelartifacts to edgeai-modelzoo/modelartifacts under ./work_dirs of this repository (remove the modelartifacts folder under ./work_dirs before that).
* While running this script, compilation of models in the model zoo will be performed as the first step before the inference. But if the pre-compiled model artifacts are present, model compilation will be skipped. 
* param.yaml file present in each model artifacts folder indicates that the model compilation is complete.
* **result.yaml file, if present in each model artifacts folder indicates that the model inference is complete. If result.yaml is present, inference is also skipped. Manually delete result.yaml if it is present (i.e. if you have done it once already) to do the inference - otherwise, the script will merely print the result information from result.yaml.**


## Compiling with a custom model or custom configuration
* To compile a custom model or a custom pipeline configuration, have look at examples custom configurations in `scripts/benchmark_custom.py`.

Launch the custom configuration by calling 
```
run_custom_pc.sh <SOC> 
```

## Generating reports
A consolidated CSV report containing all your benchmarking results can be generated by running [run_generate_report.sh](../run_generate_report.sh)
```
run_generate_report.sh 
```

## Packing the artifacts to be used for inferece in the EVM
* The imported artifacts can be used to run inference on the target device (eg. EVM)
* If you have compiled models yourself and would like to run those model artifacts in the target device, run the script 
[run_package_artifacts_for_evm.sh](../run_package_artifacts_for_evm.sh) to package the artifacts for use in the target device.
```
run_package_artifacts_for_evm.sh <SOC>
```
* These packaged artifacts can be copied to the device to run inference there.
* Please note the above comment about result.yaml. These result files are removed while packaing the artifact to be used in EVM (as we actually want to run the inference in EVM). Instead of using the packaging script, you can use the compiled model artifact directory as well, but be sure to remove the result.yaml if you actually want the inference to run.
* Change the modelartifacts_path in settings.yaml to point to the correct modelartifacts path. For example, after packaging, the packaged folder name will be <SOC>_package. To use this packaged artifacts, set modelartifacts_path accordingly.
* As explained above, once the inference is run a result.yaml file is created inside the folder for the model. Any subsequent inference will not run the actual inference, but will just report the result in that file. Delete that result.yaml file if you wish to run the inference again.


## Running inference on EVM or StarterKit board
Refer to [Instruction for running on EVM/device](./usage_evm.md)

## Debugging
The .sh scripts actually invoke python scripts in `scripts` folder - use any convenient IDE such as PyCharm or VSCode these files.

Make sure to set TIDL_TOOLS_PATH and LD_LIBRARY_PATH environment variables to `tools/<SOC>/tidl_tools` depending on your SoC - for example 'tools/AM68A/tidl_tools'.
