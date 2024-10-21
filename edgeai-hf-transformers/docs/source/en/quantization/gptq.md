<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPTQ

<Tip>

Try GPTQ quantization with PEFT in this [notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) and learn more about it's details in this [blog post](https://huggingface.co/blog/gptq-integration)!

</Tip>

The [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) library implements the GPTQ algorithm, a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes the error. These weights are quantized to int4, but they're restored to fp16 on the fly during inference. This can save your memory-usage by 4x because the int4 weights are dequantized in a fused kernel rather than a GPU's global memory, and you can also expect a speedup in inference because using a lower bitwidth takes less time to communicate.

Before you begin, make sure the following libraries are installed:

```bash
pip install auto-gptq
pip install --upgrade accelerate optimum transformers
```

To quantize a model (currently only supported for text models), you need to create a [`GPTQConfig`] class and set the number of bits to quantize to, a dataset to calibrate the weights for quantization, and a tokenizer to prepare the dataset.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

You could also pass your own dataset as a list of strings, but it is highly recommended to use the same dataset from the GPTQ paper.

```py
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
```

Load a model to quantize and pass the `gptq_config` to the [`~AutoModelForCausalLM.from_pretrained`] method. Set `device_map="auto"` to automatically offload the model to a CPU to help fit the model in memory, and allow the model modules to be moved between the CPU and GPU for quantization.

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```

If you're running out of memory because a dataset is too large, disk offloading is not supported. If this is the case, try passing the `max_memory` parameter to allocate the amount of memory to use on your device (GPU and CPU):

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"}, quantization_config=gptq_config)
```

<Tip warning={true}>

Depending on your hardware, it can take some time to quantize a model from scratch. It can take ~5 minutes to quantize the [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) model on a free-tier Google Colab GPU, but it'll take ~4 hours to quantize a 175B parameter model on a NVIDIA A100. Before you quantize a model, it is a good idea to check the Hub if a GPTQ-quantized version of the model already exists.

</Tip>

Once your model is quantized, you can push the model and tokenizer to the Hub where it can be easily shared and accessed. Use the [`~PreTrainedModel.push_to_hub`] method to save the [`GPTQConfig`]:

```py
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

You could also save your quantized model locally with the [`~PreTrainedModel.save_pretrained`] method. If the model was quantized with the `device_map` parameter, make sure to move the entire model to a GPU or CPU before saving it. For example, to save the model on a CPU:

```py
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")

# if quantized with device_map set
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

Reload a quantized model with the [`~PreTrainedModel.from_pretrained`] method, and set `device_map="auto"` to automatically distribute the model on all available GPUs to load the model faster without using more memory than needed.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

## ExLlama

[ExLlama](https://github.com/turboderp/exllama) is a Python/C++/CUDA implementation of the [Llama](model_doc/llama) model that is designed for faster inference with 4-bit GPTQ weights (check out these [benchmarks](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark)). The ExLlama kernel is activated by default when you create a [`GPTQConfig`] object. To boost inference speed even further, use the [ExLlamaV2](https://github.com/turboderp/exllamav2) kernels by configuring the `exllama_config` parameter:

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=gptq_config)
```

<Tip warning={true}>

Only 4-bit models are supported, and we recommend deactivating the ExLlama kernels if you're finetuning a quantized model with PEFT.

</Tip>

The ExLlama kernels are only supported when the entire model is on the GPU. If you're doing inference on a CPU with AutoGPTQ (version > 0.4.2), then you'll need to disable the ExLlama kernel. This overwrites the attributes related to the ExLlama kernels in the quantization config of the config.json file.

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig
gptq_config = GPTQConfig(bits=4, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="cpu", quantization_config=gptq_config)
```