import os
import sys
import copy
import warnings

try:
    import onnx
except:
    #warnings.warn('onnx could not be imported - this is not required for inference, but may be required for import')
    pass

try:
    from onnxsim import simplify
except:
    #warnings.warn('onnxsim could not be imported - this is not required for inference, but may be required for import')
    pass

from .. import utils
from .. import preprocess


def model_transformation(settings, pipeline_configs_in):
    if 'input_sizes' not in settings.model_transformation_dict:
        return
    #
    input_sizes = settings.model_transformation_dict['input_sizes']

    pipeline_configs_out = {}
    for size_id, input_size in enumerate(input_sizes):
        # modify a pipeline so that all the models use fixed input size
        # other modifications can also be defined here.
        warning_string = f'Changing input size to {input_size}.\n' \
                         f'The accuracies reported may be wrong as input_size is changed from the default value.'
        print(warning_string)

        for pipeline_id, pipeline_config_in in pipeline_configs_in.items():
            # start with fresh set of configs - not the one modified in earlier iteration
            pipeline_config = copy.deepcopy(pipeline_config_in)
            # start modifying the model for the given input resolution
            preproc_stge = pipeline_config['preprocess']
            preproc_transforms = preproc_stge.transforms
            for tidx, trans in enumerate(preproc_transforms):
                if isinstance(trans, preprocess.ImageResize):
                    trans = preprocess.ImageResize(input_size)
                    preproc_stge.set_param('resize', input_size)
                elif isinstance(trans, preprocess.ImageCenterCrop):
                    trans = preprocess.ImageCenterCrop(input_size)
                    preproc_stge.set_param('crop', input_size)
                #
                preproc_transforms[tidx] = trans
            #
            # generate temporary ONNX  model based on fixed resolution
            model_path = pipeline_config['session'].peek_param('model_path')
            # supported only for onnx
            if isinstance(model_path, (list,tuple)) or os.path.splitext(model_path)[-1] != '.onnx':
                continue
            #
            #print("=" * 64)
            #print("src model path :{}".format(model_path))

            # create a new model_id for the modified model
            new_model_id = pipeline_id[:-1] + str(1+size_id)
            pipeline_config['session'].set_param('model_id', new_model_id)

            # set model_path with the desired size so that the run_dir also will have that size
            # this step is just dummy - the model with the modified name doesn't exist at this point
            model_path_tmp = model_path.replace(".onnx", "_{}x{}.onnx".format(input_size, input_size))
            pipeline_config['session'].set_param('model_path', model_path_tmp)

            # initialize must be called to re-create run_dir and artifacts_folder for this new model_id
            # it is possible that initialize() must have been called before - we need to set run_dir to None to re-create it
            pipeline_config['session'].set_param('run_dir', None)
            pipeline_config['session'].initialize()

            # run_dir must have been created now
            run_dir = pipeline_config['session'].get_param('run_dir')
            model_folder = pipeline_config['session'].get_param('model_folder')

            # now set the final model_path
            model_path_out = os.path.join(model_folder, os.path.basename(model_path_tmp))
            pipeline_config['session'].set_param('model_path', model_path_out)

            # create the modified onnx model with the required input size
            # if the run_dir or the packaged (.tar.gz) artifact is available, this will be skipped
            tarfile_name = run_dir + '.tar.gz'
            linkfile_name = run_dir + '.tar.gz.link'
            if (not os.path.exists(run_dir)) and (not os.path.exists(tarfile_name)) and (not os.path.exists(linkfile_name)):
                onnx_model = onnx.load(model_path)
                input_name_shapes = utils.get_input_shape_onnx(onnx_model)
                assert len(input_name_shapes) == 1
                input_name = None
                for k, v in input_name_shapes.items():
                    input_name = k
                #
                out_name_shapes = utils.get_output_shape_onnx(onnx_model)

                # variable shape model
                input_var_shapes = {input_name: ['b', 3, 'w', 'h']}

                # create first varibale shape model
                onnx_model = utils.onnx_update_model_dims(onnx_model, input_var_shapes, out_name_shapes)
                input_name_shapes[input_name] = [1, 3, input_size, input_size]
                # change to fixed shape model
                try:
                    onnx_model, check = simplify(onnx_model, skip_shape_inference=False, overwrite_input_shapes=input_name_shapes)
                except Exception as e:
                    warnings.warn(f'please install onnx-simplifier : onnxsim.simplify()')
                    warnings.warn(f'changing the size of {model_path} did not work - skipping')
                    warnings.warn(f'{e}')
                    continue

                # save model in model_folder
                os.makedirs(model_folder, exist_ok=True)
                #print("saving modified model :{}".format(model_path_out))
                onnx.save(onnx_model, model_path_out)
                #onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)
            #
            pipeline_configs_out.update({new_model_id: pipeline_config})
        #
    #
    return pipeline_configs_out
