import copy
import torch.nn as nn


class FCOS3D_export_model(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 add_pred_to_datasample):
        super().__init__()
        self.backbone   = backbone.convert(make_copy=True) if hasattr(backbone, "convert") else backbone
        self.neck       = neck.convert(make_copy=True) if hasattr(neck, "convert") else neck
        if hasattr(bbox_head, "new_bbox_head"):
            self.bbox_head  = copy.deepcopy(bbox_head)
            # self.bbox_head.new_bbox_head loses the convert function after deepcopy so using the original
            setattr(self.bbox_head, "new_bbox_head", bbox_head.new_bbox_head.convert(make_copy=True))
            self.bbox_head.cpu()
        elif hasattr(backbone, "convert"): # bbox_head is not quantized but rest of the network is quantized
            self.bbox_head  = copy.deepcopy(bbox_head).cpu()
        else:
            self.bbox_head = bbox_head
        self.add_pred_to_datasample = add_pred_to_datasample

    def prepare_data(self, batch_img_metas):
        self.batch_img_metas = batch_img_metas


    def forward(self, img, pad_cam2img, inv_pad_cam2img):
        x = self.backbone(img)
        x = self.neck(x)

        outs = self.bbox_head(x)
        #return outs

        predictions = self.bbox_head.predict_by_feat_onnx(
            *outs, batch_img_metas=self.batch_img_metas, rescale=True, 
            pad_cam2img=pad_cam2img, inv_pad_cam2img=inv_pad_cam2img)

        return predictions
