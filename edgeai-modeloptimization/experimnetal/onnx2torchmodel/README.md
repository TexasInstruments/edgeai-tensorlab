# EdgeAI Model Optimization - ONNX2Torch Model Conversion

A Python toolkit for converting ONNX models to PyTorch models developed by Texas Instruments' EdgeAI, TIDL & Analytics Algo Teams.

## Overview

The `edgeai_onnx2torchmodel` package provides tools to convert ONNX models to PyTorch models. This is particularly useful for deep learning model optimization, quantization, and deployment workflows. It supports various ONNX operations and provides utilities to validate the conversion process.

## Features

- Convert ONNX models to PyTorch models
- Support for various ONNX operations
- Utilities for model validation and error checking
- Compatibility with model quantization workflows
- Support for both module-based and functional conversion
- Graph simplification and optimization

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch and TorchVision
- ONNX (version 1.16.1)
- ONNX GraphSurgeon

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeloptimization/browse
   cd edgeai-modeloptimization/experimnetal/onnx2torchmodel
   ```

2. Install using the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   Alternatively, you can install directly using pip:
   ```bash
   pip install -e .
   ```

## Usage

### Converting ONNX to PyTorch Model

```python
from edgeai_onnx2torchmodel.onnx2pytorch import convert

# Convert ONNX model to PyTorch
torch_model = convert("path/to/model.onnx")

# Use for training (if needed)
# torch_model = convert("path/to/model.onnx", for_training=True)

# Without simplification (if needed)
# torch_model = convert("path/to/model.onnx", simplify=False)
```

### ONNX to ONNX Conversion and Validation

```python
import sys
from edgeai_onnx2torchmodel.main_onnx2onnx import main

# Run conversion and validation
main(["path/to/model.onnx", "--export_txts", "--simplify"])
```

### Converting PyTorch Models

```python
import sys
from edgeai_onnx2torchmodel.main_torch2torch import main

# Convert and validate specific model
main(["resnet18", "output_directory", "--export_txts", "--simplify"])
```

## Example Workflows

### Model Conversion and Quantization

This package can be used as part of a workflow to:
1. Convert ONNX models to PyTorch
2. Apply quantization techniques
3. Export back to ONNX or deploy directly

### Error Analysis

The toolkit provides error analysis capabilities:
```python
torch_model, output1, output2, output3 = main(["path/to/model.onnx", "--export_txts"])
```

This returns:
- The converted PyTorch model
- Original model outputs
- Intermediate ONNX model outputs
- Converted PyTorch model outputs

## Command Line Interface

### ONNX to ONNX Conversion

```bash
python -m edgeai_onnx2torchmodel.main_onnx2onnx path/to/model.onnx [options]
```

Options:
- `--all_output`, `-a`: Export model with outputs of all nodes
- `--export_txts`, `-e`: Export error analysis text files
- `--for-training`, `-t`: Use training mode
- `--threshold1`, `-t1`: Threshold for absolute error (default: 1e-5)
- `--threshold2`, `-t2`: Threshold for relative error (default: 1e-2)
- `--cuda`, `-c`: Use CUDA
- `--simplify`, `-s`: Simplify model

### Torch to Torch Conversion

```bash
python -m edgeai_onnx2torchmodel.main_torch2torch model_name output_dir [options]
```

Options:
- `--simplify`, `-s`: Simplify model
- `--cuda`, `-c`: Use CUDA
- `--export_txts`, `-e`: Export error analysis text files
- `--threshold1`, `-t1`: Threshold for absolute error (default: 1e-5)
- `--threshold2`, `-t2`: Threshold for relative error (default: 1e-2)

## Contributing

Please contact the EdgeAI team at Texas Instruments for contribution guidelines and processes.

## License

Copyright (c) 2025-2026, Texas Instruments. All Rights Reserved.

This software is provided under the BSD 3-Clause License. See the LICENSE file for details.
