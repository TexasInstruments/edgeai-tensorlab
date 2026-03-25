import torch
import torchvision


MODEL_SURGERY_NAMES_LITE2ORIGINAL = {
    "deeplabv3_mobilenet_v3_large_lite": ("deeplabv3_mobilenet_v3_large", dict(group_size_dw=1)),
    "lraspp_mobilenet_v3_large_lite": ("lraspp_mobilenet_v3_large", dict(group_size_dw=1)),
}


def get_model(model_name, weights, weights_backbone, num_classes, model_surgery, aux_loss):
    surgery_kwargs = dict()
    if model_surgery and model_name in MODEL_SURGERY_NAMES_LITE2ORIGINAL:
        model_name_to_use, surgery_kwargs = MODEL_SURGERY_NAMES_LITE2ORIGINAL[model_name]
    else:
        model_name_to_use = model_name
    #
    model = torchvision.models.get_model(model_name_to_use, weights=weights, weights_backbone=weights_backbone, num_classes=num_classes, aux_loss=aux_loss)
    return model, surgery_kwargs
    