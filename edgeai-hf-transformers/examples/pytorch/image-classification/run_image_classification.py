#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from packaging.version import Version

import torch
if Version(torch.__version__.split('+')[0]) < Version('2.5'):

    import torch.ao.quantization.pt2e.utils
    from torch.export.unflatten import _assign_attr, _AttrKind
    import operator

    # monkey patch few functions of torch fixed in torch 2.5 # remove these in later releases # TODO
    def _is_supported_batch_norm_for_training_new(node):
        """
        Return True if the given node refers to an aten batch norm op QAT supports.
        """
        supported_ops = [
            torch.ops.aten.batch_norm.default,
            torch.ops.aten._native_batch_norm_legit.default,
            # Note: we won't need this op anymore after batch norm consolidation
            # For now, we need to continue to support it because it gives better
            # training numerics than `_native_batch_norm_legit`
            torch.ops.aten.cudnn_batch_norm.default,
            torch.ops.aten.miopen_batch_norm.default,
        ]
        return node.target in supported_ops

    torch.ao.quantization.pt2e.utils._is_supported_batch_norm_for_training = _is_supported_batch_norm_for_training_new

    def fold_bn_weights_into_conv_node_new_(
        conv_node,
        conv_weight_node,
        conv_bias_node,
        bn_node,
        m,
    ) -> None:
        # conv args: input, weight, bias, stride, padding, dilation, ...
        conv_w = torch.ao.quantization.pt2e.utils._get_tensor_constant_from_node(conv_weight_node, m)
        conv_b = torch.ao.quantization.pt2e.utils._get_tensor_constant_from_node(conv_bias_node, m)
        transpose = torch.ao.quantization.pt2e.utils._is_conv_transpose_node(conv_node)

        # eval bn args: input, weight, bias, running mean, running var, momentum, eps
        # train bn args: input, weight, bias, running mean, running var, training, momentum, eps
        bn_args_schema = bn_node.target._schema.arguments  # type: ignore[union-attr]
        bn_args = torch.ao.quantization.pt2e.utils._get_all_arguments(bn_node.args, bn_node.kwargs, bn_args_schema)
        bn_w = torch.ao.quantization.pt2e.utils._get_tensor_constant_from_node(bn_args[1], m)
        bn_b = torch.ao.quantization.pt2e.utils._get_tensor_constant_from_node(bn_args[2], m)
        bn_rm = torch.ao.quantization.pt2e.utils._get_tensor_constant_from_node(bn_args[3], m)
        bn_rv = torch.ao.quantization.pt2e.utils._get_tensor_constant_from_node(bn_args[4], m)
        if bn_node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
            eps_arg_index = 6
        elif torch.ao.quantization.pt2e.utils._is_supported_batch_norm_for_training(bn_node):
            eps_arg_index = 7
        else:
            raise ValueError("BN node target is unexpected ", bn_node.target)
        bn_eps = bn_args[eps_arg_index]

        fused_weight, fused_bias = torch.ao.quantization.pt2e.utils.fuse_conv_bn_weights(
            conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=transpose
        )

        # update the weight and bias for conv
        conv_args = list(conv_node.args)
        # filling in the default bias argument
        if len(conv_args) == 2:
            conv_args.append(None)

        # calling data since the fused_weight and fused_bias are nn.Parameter
        weight_attr_name = conv_weight_node.target
        assert isinstance(weight_attr_name, str)
        _assign_attr(fused_weight, m, weight_attr_name, _AttrKind.PARAMETER)
        if conv_bias_node is not None:
            bias_attr_name = conv_bias_node.target
            _assign_attr(fused_bias, m, str(bias_attr_name), _AttrKind.PARAMETER)
        else:
            bias_attr_name = weight_attr_name + "_bias"
            _assign_attr(fused_bias, m, bias_attr_name, _AttrKind.PARAMETER)
            with m.graph.inserting_before(conv_node):
                get_bias_node = m.graph.get_attr(bias_attr_name)
            # NOTE: here we assume the bias of conv is not quantized!
            conv_args[2] = get_bias_node
        conv_node.args = tuple(conv_args)

        # native_batch_norm has 3 outputs, we expect getitem calls on the output
        # and we want to replace the uses of getitem 0 with the output of conv
        #
        if bn_node.target == torch.ops.aten.batch_norm.default:
            # With the new training ir, instead of batch_norm + getitem,
            # we only have the batch_norm node.
            #
            # Before:
            # conv -> bn -> users
            # After:
            # conv -> users
            #       bn has no users now
            bn_node.replace_all_uses_with(conv_node)
        else:
            # Before:
            # conv -> bn - (first output) -> users1
            #          \ - (second output) -> users2
            #          \ - (third output) -> users3
            # After:
            # conv -> (first output) -> users1
            #       bn -
            #          \ - (second output) -> users2
            #          \ - (third output) -> users3
            # if users2 and users3 are empty then bn will be removed through dead code elimination
            for user in bn_node.users:
                if (
                    user.op != "call_function"
                    or user.target != operator.getitem
                    or user.args[1] != 0
                ):
                    continue
                user.replace_all_uses_with(conv_node)

        # If the BN node does not have users, erase it from the graph
        # Note: we need to do this manually because the model can still be in train
        # mode at this point, in which case DCE won't erase the BN node automatically
        # since the node refers to a mutating op. Here we still need to call DCE first
        # to get rid of the unused getitem nodes that consume the BN node.
        m.graph.eliminate_dead_code()
        if len(bn_node.users) == 0:
            m.graph.erase_node(bn_node)

    def _fuse_conv_bn_new_(m) -> None:
        has_bn = any(torch.ao.quantization.pt2e.utils._is_bn_node(n) for n in m.graph.nodes)
        if not has_bn:
            return
        for n in m.graph.nodes:
            if n.op != "call_function" or n.target not in (torch.ops.aten._native_batch_norm_legit_no_training.default, torch.ops.aten.batch_norm.default):
                continue
            bn_node = n
            n = bn_node.args[0]
            if not torch.ao.quantization.pt2e.utils._is_conv_or_conv_transpose_node(n):
                continue
            conv_node = n
            conv_weight_node = conv_node.args[1] 
            if conv_weight_node.op != "get_attr":    ################### this is not fixed in later releases as well, ensure this is kept ###################
                conv_weight_node = conv_weight_node.args[0] # weights are coming after reshape
            conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
            fold_bn_weights_into_conv_node_new_(conv_node, conv_weight_node, conv_bias_node, bn_node, m)

        m.graph.eliminate_dead_code()
        m.recompile()

    torch.ao.quantization.pt2e.utils._fuse_conv_bn_= _fuse_conv_bn_new_


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    TimmWrapperImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import copy

present_torchmodelopt = True
try:
    from edgeai_torchmodelopt import xmodelopt
except ModuleNotFoundError:
    present_torchmodelopt = False

torch.backends.cuda.enable_mem_efficient_sdp(False) # disabling the efficient attention block to support model onnx export

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub or the path of the dataset (should have the loading file) \
                (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    loading_num_proc: Optional[int] = field(
        default=8, metadata={"help": "Sharding of the dataset loading to make it faster"}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    size: Optional[str] = field(
        default=None,
        metadata={"help": "Image resize - it it is an int, resize the shortest edge to this size."},
    )
    crop_size: Optional[str] = field(
        default=None,
        metadata={"help": "Image crop size - center crop to this size."},
    )
    rescale_factor: Optional[float] = field(
        default=1/255,
        metadata={"help": "rescale_factor to multiply the input image."},
    )
    image_mean: Optional[str] = field(
        default=None,
        metadata={"help": "Mean value to be subtracted from input image."},
    )
    image_std: Optional[str] = field(
        default=None,
        metadata={"help": "Std to be used to divide the input image."},
    )
    image_scale: Optional[str] = field(
        default=None,
        metadata={"help": "Scale value to multiply the input image."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class ModelOptimizationArguments:
    """
    Arguments pertaining to which type of model optimizations needs to be done.
    """
    
    model_surgery: int = field(
        default=0, 
        metadata={
            "help": "Whether we need to do model surgery in our network. (Options. 0(No Surgery), 1(native), 2(fx), 3(pt2e))"
    })
    quantization: int = field(
        default=0,
        metadata={
            "help" : "Whether we need to introduce quantization in our network. (Options. 0(No Surgery), 1(native), 2(fx), 3(pt2e)). \
                As of now, only 0/3 is supported for this repository"
        }
    )
    quantize_type: str = field(
        default='QAT',
        metadata={
            "help" : "How do we want to quantize our network (Options. QAT, PTC/PTQ). This is only applicable when quantization is set to 3"
        }
    )
    qconfig_type: str = field(
        default='DEFAULT',
        metadata={
            "help" : "The qconfig schemes for inducing the quantization. (Options. DEFAULT, WC8_AT8, MSA_WC8_AT8, WT8SYMP2_AT8SYMP2, ...). \
                The full list and explanations can be obtained from qconfig_types.py"
        }
    )
    quantize_calib_images: int = field(
        default=50,
        metadata={
            "help" : "Incase of PTC/PTQ, the number of images to be used for calibration"
        }
    )
    bias_calibration_factor: float = field(
        default=0,
        metadata={
            "help" : "The bias calibration factor to be used, 0 incase of no bias calibration "
        }
    )
    do_onnx_export: bool = field(
        default=True,
        metadata={
            "help": "Whether we want to export the onnx network. (Default=True)"
        }
    )
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ModelOptimizationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, model_optimization_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, model_optimization_args = parser.parse_args_into_dataclasses()

    if model_args.size is not None:
        model_args.size = [int(word) for word in model_args.size.split(" ")]
        if len(model_args.size) == 1:
            model_args.size = {"shortest_edge": model_args.size[0]}

    if model_args.crop_size is not None:
        model_args.crop_size = [int(word) for word in model_args.crop_size.split(" ")]
        if len(model_args.crop_size) == 1:
            model_args.crop_size = (model_args.crop_size[0], model_args.crop_size[0])

    if model_args.image_std and model_args.image_scale:
        assert False, "only one of image_std or image_scale should be specified"
    elif model_args.image_scale is not None:
        model_args.image_std = [1/float(word) for word in model_args.image_scale.split(" ")]
    elif model_args.image_std is not None:
        model_args.image_std = [float(word) for word in model_args.image_std.split(" ")]

    if model_args.image_mean is not None:
        model_args.image_mean = [float(word) for word in model_args.image_mean.split(" ")]

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            num_proc=data_args.loading_num_proc
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features[data_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    # metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    metric = evaluate.load("evaluate/metrics/accuracy/accuracy.py", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="eager", # the default is sdpa mode, need to use eager for proper quantization
        return_dict=None # for prepare_qat_pt2e() used in quantization
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        rescale_factor=model_args.rescale_factor,
        size=model_args.size, 
        crop_size=model_args.crop_size,
        image_mean=model_args.image_mean,
        image_std=model_args.image_std
    )

    # Define torchvision transforms to be applied to each image.
    if isinstance(image_processor, TimmWrapperImageProcessor):
        _train_transforms = image_processor.train_transforms
        _val_transforms = image_processor.val_transforms
    else:
        if model_args.size is not None:
            size = model_args.size["shortest_edge"] if "shortest_edge" in model_args.size else model_args.size
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

        if model_args.crop_size is not None:
            crop_size = model_args.crop_size
        elif not hasattr(image_processor, 'crop_size'):
            crop_size = size
        elif isinstance(image_processor.crop_size, (int,tuple)):
            crop_size = image_processor.crop_size
        elif "shortest_edge" in image_processor.crop_size:
            crop_size = image_processor.crop_size["shortest_edge"]
        else:
            crop_size = (image_processor.crop_size["height"], image_processor.crop_size["width"])

        # size if is a int, then it is supposed to be the shortest_edge
        # crop_size is a tuple of expected final dimension
    
        # image_std = model_args.image_std or (image_processor.image_std if hasattr(image_processor, "image_std") else None)  
        # image_mean = model_args.image_mean or (image_processor.image_mean if hasattr(image_processor, "image_mean") else None)

        normalize = (
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
            if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
            else Lambda(lambda x: x)
        )
        _train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                Lambda(lambda x: np.array(x, dtype=np.float32)*model_args.rescale_factor), # to avoid division by 255 in to_tensor, handle it here
                ToTensor(),
                normalize,
            ]
        )
        _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                Lambda(lambda x: np.array(x, dtype=np.float32)*model_args.rescale_factor), # to avoid division by 255 in to_tensor, handle it here     
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        del example_batch[data_args.image_column_name]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        del example_batch[data_args.image_column_name]
        return example_batch

    if model_optimization_args.quantization and not(present_torchmodelopt):
        raise Exception("TorchmodelOpt not present and requesting quantization.") 

    if data_args.max_train_samples is None and model_optimization_args.quantization and model_optimization_args.quantize_type == "PTQ":
        data_args.max_train_samples = model_optimization_args.quantize_calib_images * training_args.per_device_eval_batch_size * training_args.n_gpu

    assert (model_optimization_args.quantization==0 or model_optimization_args.quantization==3), \
        print("Only pt2e (args.quantization=3) based quantization is currently supported for hf-transformers ")

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    if model_optimization_args.quantization == 3:
        assert training_args.per_device_train_batch_size == training_args.per_device_eval_batch_size, \
            print("only fixed batch size across train and eval is currently supported, (args.per_device_train_batch_size and args.per_device_eval_batch_size should be same)")  
        example_kwargs = next(iter(dataset["validation"]))
        example_kwargs['labels'] = torch.tensor(example_kwargs.pop('label')).unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1)
        example_input= example_kwargs.pop('pixel_values').unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
        prepare_device = None

        if not training_args.use_cpu:
            prepare_device = 'cuda'

        # epochs update in the quantization with every model.train() call, thus using the observer and batchnorm update accordingly
        # in this code model.train() model.eval() is switched for every batch / iteration - we need to set the total_epochs passed to QAT/PTQ model accordingly
        # alternate way is to specify num_observer_update_epochs num_batch_norm_update_epochs to reflect the iterations instead of epochs:   
        # num_observer_update_epochs = int(len(dataset["train"]) * ((training_args.num_train_epochs//2)+1) / (training_args.n_gpu*training_args.per_device_train_batch_size))
        # num_batch_norm_update_epochs = int(len(dataset["train"]) * ((training_args.num_train_epochs//2)-1) / (training_args.n_gpu*training_args.per_device_train_batch_size))
        total_iterations = int(training_args.num_train_epochs * len(dataset["train"]) / (training_args.n_gpu*training_args.per_device_train_batch_size))
        if model_optimization_args.quantize_type == "QAT":
            model = xmodelopt.quantization.v3.QATPT2EModule(model, total_epochs=total_iterations, is_qat=True, fast_mode=False,
                qconfig_type=model_optimization_args.qconfig_type, example_inputs=example_input, example_kwargs=example_kwargs, 
                bias_calibration_factor=model_optimization_args.bias_calibration_factor, device=prepare_device)
        else: 
            model = xmodelopt.quantization.v3.QATPT2EModule(model, total_epochs=total_iterations, is_qat=False, fast_mode=False,
                qconfig_type=model_optimization_args.qconfig_type, example_inputs=example_input, example_kwargs=example_kwargs, 
                bias_calibration_factor=model_optimization_args.bias_calibration_factor, device=prepare_device)
            # need to turn the parameter update off during PTQ/PTC
            training_args.dont_update_parameters = True
        #
    #

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
    # Model ONNX Export
    if model_optimization_args.do_onnx_export:
        export_device = 'cpu'
        file_name = model_args.model_name_or_path.split("/")[-1]
        file_name = training_args.output_dir + '/' + file_name + '_quantized.onnx' if model_optimization_args.quantization else \
            training_args.output_dir + '/' + file_name + '.onnx'
        if hasattr(trainer.model, 'export'):
            example_input = next(iter(dataset["validation"]))
            example_input['labels'] = torch.tensor(example_input.pop('label')).unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1)
            example_input['pixel_values'] = example_input['pixel_values'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
            trainer.model.export(example_input, filename=file_name, simplify=True, device=export_device)
        else:
            export_model = copy.deepcopy(trainer.model.eval()).to(device=export_device)
            example_input = next(iter(dataset["validation"]))
            labels = example_input.pop('label')
            # example_input['labels'] = torch.tensor(example_input.pop('label')).unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1)
            example_input['pixel_values'] = example_input['pixel_values'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
            if isinstance(example_input, dict):
                example_inputs = ()
                for val in example_input.values():
                    example_inputs += tuple([val.to(device=export_device)])
            else:
                example_inputs = example_input.to(device=export_device)
            torch.onnx.export(export_model, example_inputs, file_name, opset_version=17, training=torch._C._onnx.TrainingMode.PRESERVE)
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(file_name)
            onnx_model, check = simplify(onnx_model)
            onnx.save(onnx_model, file_name)
            
        print("Model Export is now complete! \n")
       
    # if model_optimization_args.quantization:
    #     trainer.model = trainer.model.convert(device='cuda')

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
