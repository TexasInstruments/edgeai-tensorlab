# Pre-Complied Model Artifacts

Models and pre-compied model artifacts are provided in another repository called **[edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo)**.

Please clone that repository. After cloning, edgeai-benchmark and edgeai-modelzoo must be inside the same parent folder for the default settings to work.

By default, modelartifacts_path in [settings_base.yaml](../settings_base.yaml) to points to a folder within this repository. To use the pretrained modelartifacts in edgeai-modelzoo, change it to modelartifacts folder inside edgeai-modelzoo. For example:
```
modelartifacts_path : '../edgeai-modelzoo/modelartifacts/{target_device}'
```

Note: {target_device} will be replaced by target_device specified as argument to [run_benchmarks_pc.sh](../run_benchmarks_pc.sh) or  [run_benchmarks_evm.sh](../run_benchmarks_evm.sh)
