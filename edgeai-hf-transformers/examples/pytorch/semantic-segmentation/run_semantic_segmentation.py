#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import albumentations as A
import evaluate
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from edgeai_torchmodelopt import xmodelopt

""" Finetuning any 🤗 Transformers model supported by AutoModelForSemanticSegmentation for semantic segmentation leveraging the Trainer API."""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.44.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/semantic-segmentation/requirements.txt")


def reduce_labels_transform(labels: np.ndarray, **kwargs) -> np.ndarray:
    """Set `0` label as with value 255 and then reduce all other labels by 1.

    Example:
        Initial class labels:         0 - background; 1 - road; 2 - car;
        Transformed class labels:   255 - background; 0 - road; 1 - car;

    **kwargs are required to use this function with albumentations.
    """
    labels[labels == 0] = 255
    labels = labels - 1
    labels[labels == 254] = 255
    return labels


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="segments/sidewalk-semantic",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
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
    do_reduce_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to reduce all labels by 1 and replace background by 255."},
    )
    reduce_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to reduce all labels by 1 and replace background by 255."},
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )
        if self.reduce_labels:
            self.do_reduce_labels = self.reduce_labels
            warnings.warn(
                "The `reduce_labels` argument is deprecated and will be removed in v4.45. Please use `do_reduce_labels` instead.",
                FutureWarning,
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="nvidia/mit-b0",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
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
    send_example_telemetry("run_semantic_segmentation", model_args, data_args)

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

    # Load dataset
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # TODO support datasets from local folders
    dataset = load_dataset(
        data_args.dataset_name, data_dir=data_args.dataset_name, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    )

    # Rename column names to standardized names (only "image" and "label" need to be present)
    if "pixel_values" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"pixel_values": "image"})
    if "annotation" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"annotation": "label"})

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    if data_args.dataset_name == "scene_parse_150":
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
    else:
        repo_id = data_args.dataset_name
        filename = "id2label.json"
    id2label = json.load(open(os.path.join(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: str(k) for k, v in id2label.items()}

    # Load the mean IoU metric from the evaluate package
    metric = evaluate.load("mean_iou", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    @torch.no_grad()
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=False, # The labels have already been reduced once during the loading, need not be reduced again
        )
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i+1]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i+1]}": v for i, v in enumerate(per_category_iou)})
        return metrics

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        do_reduce_labels=data_args.do_reduce_labels,
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
        
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    if isinstance(size, int):
        height=width=size
    else:
        height = size[0]
        width = size[1]
    
    # size if is a int, then it is supposed to be the shortest_edge
    # crop_size is a tuple of expected final dimension
    
    train_transforms = A.Compose(
        [
            A.Lambda(
                name="reduce_labels",
                mask=reduce_labels_transform if data_args.do_reduce_labels else None,
                p=1.0,
            ),
            # pad image with 255, because it is ignored by loss
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=255, p=1.0),
            A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std, max_pixel_value=(1/image_processor.rescale_factor), p=1.0),
            ToTensorV2(),
        ]
    )
    
    if isinstance(size, tuple):
        transform = [A.Resize(height=size[0], width=size[1], p=1.0)]
    else:
        transform = [
            A.SmallestMaxSize(max_size=size, p=1.0),
            A.CenterCrop(height=crop_size[0], width=crop_size[1], p=1.0),
        ]
        
    val_transforms = A.Compose(
        [
            A.Lambda(
                name="reduce_labels",
                mask=reduce_labels_transform if data_args.do_reduce_labels else None,
                p=1.0,
            ),
            *transform,
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std, max_pixel_value=(1/image_processor.rescale_factor), p=1.0),
            ToTensorV2(),
        ]
    )

    def preprocess_batch(example_batch, transforms: A.Compose):
        pixel_values = []
        labels = []
        for image, target in zip(example_batch["image"], example_batch["label"]):
            transformed = transforms(image=np.array(image.convert("RGB")), mask=np.array(target))
            pixel_values.append(transformed["image"])
            labels.append(transformed["mask"])

        encoding = {}
        encoding["pixel_values"] = torch.stack(pixel_values).to(torch.float)
        encoding["labels"] = torch.stack(labels).to(torch.long)

        return encoding

    # Preprocess function for dataset should have only one argument,
    # so we use partial to pass the transforms
    preprocess_train_batch_fn = partial(preprocess_batch, transforms=train_transforms)
    preprocess_val_batch_fn = partial(preprocess_batch, transforms=val_transforms)

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
        dataset["train"].set_transform(preprocess_train_batch_fn)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(preprocess_val_batch_fn)


    if model_optimization_args.quantization == 3:
        assert training_args.per_device_train_batch_size == training_args.per_device_eval_batch_size, \
            print("only fixed batch size across train and eval is currently supported, (args.per_device_train_batch_size and args.per_device_eval_batch_size should be same)")  
        example_input = next(iter(dataset["validation"]))
        example_input['labels'] = example_input['labels'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1)
        example_input['pixel_values'] = example_input['pixel_values'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
        convert_to_cuda = False if training_args.use_cpu else True
        
        if model_optimization_args.quantize_type == "QAT":
            num_observer_update_epochs = int(len(dataset["train"]) * ((training_args.num_train_epochs//2)+1) / (training_args.n_gpu*training_args.per_device_train_batch_size))
            num_batch_norm_update_epochs = int(len(dataset["train"]) * ((training_args.num_train_epochs//2)-1) / (training_args.n_gpu*training_args.per_device_train_batch_size))
            model = xmodelopt.quantization.v3.QATPT2EModule(model, total_epochs=training_args.num_train_epochs, is_qat=True, fast_mode=False,
                qconfig_type="DEFAULT", example_inputs=example_input, convert_to_cuda=convert_to_cuda, 
                bias_calibration_factor=model_optimization_args.bias_calibration_factor,
                num_observer_update_epochs = num_observer_update_epochs,
                num_batch_norm_update_epochs = num_batch_norm_update_epochs)
        #
        
        else: 
            # training_args.num_train_epochs = 2 # bias calibration in the second epoch
            model = xmodelopt.quantization.v3.QATPT2EModule(model, total_epochs=training_args.num_train_epochs, is_qat=True, fast_mode=False,
                qconfig_type="DEFAULT", example_inputs=example_input, convert_to_cuda=convert_to_cuda, 
                bias_calibration_factor=model_optimization_args.bias_calibration_factor, 
                num_observer_update_epochs=model_optimization_args.quantize_calib_images)
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
        tokenizer=image_processor,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Model ONNX Export
    if model_optimization_args.do_onnx_export:
        export_device = 'cpu'
        original_device = next(trainer.model.parameters()).device
        file_name = model_args.model_name_or_path.split("/")[-1]
        file_name = training_args.output_dir + '/' + file_name + '_quantized.onnx' if model_optimization_args.quantization else \
            training_args.output_dir + '/' + file_name + '.onnx'
        if hasattr(trainer.model, 'export'):
            trainer.model.export(example_input, filename=file_name, simplify=True, device=export_device)
        else:
            export_model = trainer.model.eval().to(export_device)
            example_input = next(iter(dataset["validation"]))
            labels = example_input.pop('labels')
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
            trainer.model = trainer.model.to(original_device)
            
        print("Model Export is now complete! \n")
       
    if model_optimization_args.quantization:
        trainer.model = trainer.model.convert()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["image-segmentation", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
