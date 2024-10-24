import  types
from .hooks import register_pre_hook_for_optimization, disable_pre_hook_for_optimization
from . import TransformationWrapper  


def wrap_fn_for_replace_PE(fn, module, *args, **kwargs):
# replace the specific part of the network with a constant, that part need to be traced separately, and the captured args and kwargs are used 
# to obtain the value of the module. This is done in case of specific proprocessings, that might not even be needed in the case of training, 
# for example, sine positional embeddings (not learned) which are dependant on the input image size and thus a part of the network.
    if hasattr(module, 'constant_PE'):
        return module
    else:
        if fn in (register_pre_hook_for_optimization, disable_pre_hook_for_optimization):
            module = fn(module, *args, **kwargs)
        else:
            assert hasattr(module, '_example_inputs') 
            val = module(*module._example_inputs, **module._example_kwargs)
            module.register_buffer('constant_PE', val)
            def new_forward(self, *args, **kwargs) -> tuple[list]:
                return self.constant_PE
            module.forward = types.MethodType(new_forward, module)
    return module

def DETR_transformation():
    # transformation dict to be used for quantizating the DETR architecture.
    transformation_dict = dict(class_labels_classifier=None, bbox_predictor=None)
    transformation_dict["model.backbone.position_embedding"] = TransformationWrapper(wrap_fn_for_replace_PE)
    transformation_dict["model.backbone.conv_encoder"] = None
    transformation_dict["model.input_projection"] = None
    transformation_dict["model.encoder"] = None
    transformation_dict["model.decoder"] = None
    return transformation_dict


transformation_mapping = {
    "transformers_DETR" : DETR_transformation()
}

def get_transformation_for_model(model_name):
    if model_name not in transformation_mapping.keys():
        raise Exception("Transformation mapping for {} is not defined.".format(model_name))
    return transformation_mapping[model_name]