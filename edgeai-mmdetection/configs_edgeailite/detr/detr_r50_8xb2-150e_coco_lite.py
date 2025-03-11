_base_ = [
    '../../configs/detr/detr_r50_8xb2-150e_coco.py'
]
model = dict(
    decoder=dict(  # DetrTransformerDecoder
        return_intermediate=False))