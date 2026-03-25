import torch
import torch.distributed
import onnx
from onnxsim import simplify
from  mmengine.dist.utils import  master_only

from .onnx_network import FCOS3D_export_model



@master_only
def export_FCOS3D(model, inputs=None, data_samples=None, quantized_model=False, opset_version=17):
    onnxModel = FCOS3D_export_model(model.backbone,
                                    model.neck,
                                    model.bbox_head,
                                    model.add_pred_to_datasample)
    
    if not quantized_model:
        onnxModel.eval()

    # Should clone. Otherwise, when we run both export_model and self.predict,
    # we have error PETRHead forward() - Don't know why
    img = inputs['imgs'].clone()
    batch_img_metas = [ds.metainfo for ds in data_samples]

    onnxModel.prepare_data(batch_img_metas)

    cam2img = torch.Tensor(batch_img_metas[0]['cam2img'])
    pad_cam2img = torch.eye(4, dtype=cam2img.dtype).cuda()
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = pad_cam2img.inverse().transpose(0, 1)

    if quantized_model:
        modelInput = []
        modelInput.append(img.cpu())
        modelInput.append(pad_cam2img.cpu())
        modelInput.append(inv_pad_cam2img.cpu())

        from edgeai_torchmodelopt import xmodelopt
        xmodelopt.quantization.v3.quant_utils.register_onnx_symbolics(opset_version=opset_version)

        model_name = 'fcos3d_quantized.onnx'
    else:
        modelInput = []
        modelInput.append(img)
        modelInput.append(pad_cam2img)
        modelInput.append(inv_pad_cam2img)

        model_name = 'fcos3d.onnx'

    # Save input & output images
    #fcos3d_img_np  = img.to('cpu').numpy()
    #fcos3d_img_np.tofile('fcos3d_img_np.dat')
    #out = onnxModel(img)
    #for i in range(len(out)):
    #    for j in range(len(out[i])):
    #        out[i][j].to('cpu').numpy().tofile(f"fcos3d_out_{i}_{j}.dat")

    input_names  = ["inputs", "pad_cam2img", "inv_pad_cam2img"]
    output_names = ["mlvl_bboxes", "mlvl_bboxes_for_nms", "mlvl_nms_scores", 
                    "mlvl_dir_scores", "mlvl_attr_scores"]


    torch.onnx.export(onnxModel,
                      tuple(modelInput),
                      model_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=opset_version,
                      training=torch._C._onnx.TrainingMode.PRESERVE,
                      do_constant_folding=False,
                      verbose=False)

    onnx_model, _ = simplify(model_name)
    onnx.save(onnx_model, model_name)

    print("!! ONNX model has been exported for FCOS3D! !!\n\n")
