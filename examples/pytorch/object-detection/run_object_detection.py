#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Finetuning any 🤗 Transformers model supported by AutoModelForObjectDetection for object detection leveraging the Trainer API."""

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from edgeai_torchmodelopt import xmodelopt

torch.backends.cuda.enable_mem_efficient_sdp(False) # disabling the efficient attention block to support model onnx export


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/object-detection/requirements.txt")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: str = field(
        default="cppe-5",
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
    image_square_size: Optional[int] = field(
        default=600,
        metadata={"help": "Image longest size will be resized to this value, then image will be padded to square."},
    )
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set. If value<1, use it as a percentage of total training examples"
            )
        },
    )
    max_eval_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set.  If value<1, use it as a percentage of total validation examples"
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/detr-resnet-50",
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
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model (if for instance, you are instantiating a model with 10 labels from a checkpoint with 3 labels)."
        },
    )
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
            "help": "Whether we need to do model surgery in our network. (Options. 0(No Surgery), 1(native), 2(fx/pt2e))"
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
    qconfig_type: str = field(
        default='DEFAULT',
        metadata={
            "help" : "The qconfig schemes for inducing the quantization. (Options. DEFAULT(MSA_WC8_AT8), WC8_AT8, MSA_WC8_AT8, WT8SYMP2_AT8SYMP2, ...). \
                The full list and explanations can be obtained from qconfig_types.py"
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

    if model_args.image_std and model_args.image_scale:
        assert False, "only one of image_std or image_scale should be specified"
    elif model_args.image_scale is not None:
        model_args.image_std = [1/float(word) for word in model_args.image_scale.split(" ")]
    elif model_args.image_std is not None:
        model_args.image_std = [float(word) for word in model_args.image_std.split(" ")]

    if model_args.image_mean is not None:
        model_args.image_mean = [float(word) for word in model_args.image_mean.split(" ")]
          
    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_object_detection", model_args, data_args)

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
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(
        data_args.dataset_name, data_dir=data_args.dataset_name, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    )

    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Get dataset categories and prepare mappings for label_name <-> label_id
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------

    common_pretrained_args = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        attn_implementation="eager", # the default is sdpa mode, need to use eager for proper quantization
        return_dict=None, # for prepare_qat_pt2e() used in quantization
        **common_pretrained_args,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **common_pretrained_args,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        **common_pretrained_args,
    )
    
    if model_args.image_mean is not None:
        image_processor.image_mean = model_args.image_mean
    if model_args.image_std is not None:
        image_processor.image_std = model_args.image_std
    if model_args.rescale_factor is not None:
        image_processor.rescale_factor = model_args.rescale_factor
    
    # Define torchvision transforms to be applied to each image.
    if model_args.size is not None:
        image_processor.size['height'] = model_args.size["shortest_edge"] if "shortest_edge" in model_args.size else model_args.size
        image_processor.size['width'] = model_args.size["shortest_edge"] if "shortest_edge" in model_args.size else model_args.size
        if "shortest_edge" in image_processor.size:
            temp = image_processor.size.pop("shortest_edge")
        if "longest_edge" in image_processor.size:
            temp = image_processor.size.pop("longest_edge")
        if isinstance(image_processor.size['height'], tuple):
            image_processor.size['height'] = image_processor.size['height'][0]
            image_processor.size['width'] = image_processor.size['width'][1]
    
    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
    max_size = data_args.image_square_size
    train_augment_and_transform = A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=max_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor, return_pixel_mask=True
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor, return_pixel_mask=True
    )
    
    if data_args.max_train_samples is None and model_optimization_args.quantization and model_optimization_args.quantize_type in ["PTQ","PTC"]:
        data_args.max_train_samples = model_optimization_args.quantize_calib_images * training_args.per_device_eval_batch_size * training_args.n_gpu

    assert (model_optimization_args.quantization==0 or model_optimization_args.quantization==3), \
        print("Only pt2e (args.quantization=3) based quantization is currently supported for hf-transformers ")
        
    if data_args.max_train_samples is not None and data_args.max_train_samples < 1:
        data_args.max_train_samples = len(dataset["train"])*data_args.max_train_samples
    if data_args.max_eval_samples is not None and data_args.max_eval_samples < 1:
        data_args.max_eval_samples = len(dataset["validation"])*data_args.max_eval_samples

    dataset["train"] = dataset["train"].with_transform(train_transform_batch) if data_args.max_train_samples is None else \
        dataset["train"].with_transform(train_transform_batch).select(range(int(data_args.max_train_samples)))
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch) if data_args.max_eval_samples is None else \
        dataset["validation"].with_transform(validation_transform_batch).select(range(int(data_args.max_eval_samples)))
    # dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    assert (model_optimization_args.quantization==0 or model_optimization_args.quantization==3), \
        print("Only pt2e (args.quantization=3) based quantization is currently supported for hf-transformers")

    
    if model_optimization_args.quantization == 3:
        print("The quantization flow is currently not working with this flow.\n")
        quit()
        assert training_args.per_device_train_batch_size == training_args.per_device_eval_batch_size, \
            print("only fixed batch size across train and eval is currently supported, (args.per_device_train_batch_size and args.per_device_eval_batch_size should be same)")  
        example_kwargs = next(iter(dataset["validation"]))
        example_kwargs['pixel_values'] = example_kwargs['pixel_values'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
        pixel_mask = example_kwargs.pop('pixel_mask')
        # example_kwargs['pixel_mask'] = example_kwargs['pixel_mask'].repeat(training_args.per_device_train_batch_size, 1, 1)
        example_kwargs['labels'] = [example_kwargs.pop('labels')]*training_args.per_device_train_batch_size
        convert_to_cuda = False if training_args.use_cpu else True
        
        from edgeai_torchmodelopt.xmodelopt import get_transformation_for_model
        transformation_dict = get_transformation_for_model("transformers_DETR")
        copy_attrs = []
        example_inputs = []
        
        if model_optimization_args.quantize_type == "QAT":
            num_observer_update_epochs = int(len(dataset["train"]) * ((training_args.num_train_epochs//2)+1) / (training_args.n_gpu*training_args.per_device_train_batch_size))
            num_batch_norm_update_epochs = int(len(dataset["train"]) * ((training_args.num_train_epochs//2)-1) / (training_args.n_gpu*training_args.per_device_train_batch_size))
            quantization_kwargs = dict(quantization_method='QAT', total_epochs=training_args.num_train_epochs, qconfig_type=model_optimization_args.qconfig_type, convert_to_cuda=convert_to_cuda, 
                                bias_calibration_factor=model_optimization_args.bias_calibration_factor, num_observer_update_epochs = num_observer_update_epochs,
                                num_batch_norm_update_epochs = num_batch_norm_update_epochs)
            
            model = xmodelopt.apply_model_optimization(model, example_inputs, example_kwargs, model_surgery_version=0, \
                                quantization_version=model_optimization_args.quantization, model_surgery_kwargs=None, \
                                quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)
            
        else:
            data_args.max_train_samples = model_optimization_args.quantize_calib_images * training_args.per_device_eval_batch_size * training_args.n_gpu
            training_args.num_train_epochs = 2 # bias calibration in the second epoch
            quantization_kwargs = dict(quantization_method='PTC', total_epochs=training_args.num_train_epochs, qconfig_type=model_optimization_args.qconfig_type, convert_to_cuda=convert_to_cuda, 
                                bias_calibration_factor=model_optimization_args.bias_calibration_factor, num_observer_update_epochs = model_optimization_args.quantize_calib_images )
            
            model = xmodelopt.apply_model_optimization(model, example_inputs, example_kwargs, model_surgery_version=0, \
                                quantization_version=model_optimization_args.quantization, model_surgery_kwargs=None, \
                                quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)
            
            # need to turn the parameter update off during PTQ/PTC
            training_args.dont_update_parameters = True
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn
    )

    # Training
    if training_args.do_train:
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
            example_kwargs = next(iter(dataset["validation"]))
            example_kwargs['pixel_values'] = example_kwargs['pixel_values'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
            # example_kwargs['pixel_mask'] = example_kwargs['pixel_mask'].repeat(training_args.per_device_train_batch_size, 1, 1)
            pixel_mask = example_kwargs.pop('pixel_mask')
            example_kwargs['labels'] = [example_kwargs.pop('labels')]*training_args.per_device_train_batch_size
            # maybe labels are required being a traced model, see to it
            trainer.model.export(example_kwargs, filename=file_name, simplify=True, device=export_device, make_copy=True)
        else:
            trainer.model.eval().to(device=export_device)
            example_kwargs = next(iter(dataset["validation"]))
            pixel_values = example_kwargs['pixel_values'].unsqueeze(0).repeat(training_args.per_device_train_batch_size, 1, 1, 1)
            torch.onnx.export(trainer.model, pixel_values, file_name, opset_version=17, training=torch._C._onnx.TrainingMode.PRESERVE)
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(file_name)
            onnx_model, check = simplify(onnx_model)
            onnx.save(onnx_model, file_name)
            
        print("Model Export is now complete! \n")
    
    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
