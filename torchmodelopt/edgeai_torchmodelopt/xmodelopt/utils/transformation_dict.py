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


def wrap_fn_for_bbox_head(fn, module, *args, **kwargs):
    # modify the forward function of bbox_head to call the new_bbox_head and do all the model optimizations on the new model
    if hasattr(module, 'new_bbox_head'):
        module.new_bbox_head = fn(module.new_bbox_head, *args, **kwargs)
    else:
        new_bbox_head = fn(module, *args, **kwargs)
        if new_bbox_head is not module:    
            module.add_module('new_bbox_head', new_bbox_head)
            def new_forward(self, x: tuple[torch.Tensor]) -> tuple[list]:
                return self.new_bbox_head(x)
            module.forward = types.MethodType(new_forward, module)
            if isinstance(new_bbox_head, GraphModule):
                params = dict(module.named_parameters())
                for key in params:
                    if key.startswith("new_bbox_head."):
                        continue
                    split = key.rsplit('.',1)
                    if len(split) == 1:
                        param_name = split[0]
                        delattr(module, param_name)
                    else:
                        parent_module, param_name = split
                        main_module = parent_module.split('.',1)[0]
                        if hasattr(module, main_module):
                            delattr(module, main_module)
        else:
            module = new_bbox_head
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