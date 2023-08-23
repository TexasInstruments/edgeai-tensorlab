import torch
import torchvision


MODEL_SURGERY_NAMES_LITE2ORIGINAL = {
    "mobilenet_v2_lite": ("mobilenet_v2", dict()),
    "mobilenet_v3_large_lite": ("mobilenet_v3_large", dict()),
    "mobilenet_v3_small_lite": ("mobilenet_v3_small", dict()),
}


def get_model(model_name, weights, num_classes, model_surgery):
    surgery_kwargs = dict()
    if model_surgery and model_name in MODEL_SURGERY_NAMES_LITE2ORIGINAL:
        model_name_to_use, surgery_kwargs = MODEL_SURGERY_NAMES_LITE2ORIGINAL[model_name]
    else:
        model_name_to_use = model_name
    #
    model = torchvision.models.get_model(model_name_to_use, weights=weights, num_classes=num_classes)
    return model, surgery_kwargs
