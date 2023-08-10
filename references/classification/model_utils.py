import torch
import torchvision


MODEL_SURGERY_NAMES_ORIGINAL2LITE = {
    "mobilenet_v2": "mobilenet_v2_lite"
}


MODEL_SURGERY_NAMES_LITE2ORIGINAL = {v:k for k,v in MODEL_SURGERY_NAMES_ORIGINAL2LITE.items()}


def get_model(model_name, weights, num_classes, model_surgery):
    if model_surgery and model_name in MODEL_SURGERY_NAMES_LITE2ORIGINAL:
        model_name_to_use = MODEL_SURGERY_NAMES_LITE2ORIGINAL[model_name]
    else:
        model_name_to_use = model_name
    #
    model = torchvision.models.get_model(model_name_to_use, weights=weights, num_classes=num_classes)
    return model
