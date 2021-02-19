import cv2
from jacinto_ai_benchmark import *


def get_configs(settings, work_dir,
                onnx_session_type=sessions.TVMDLRSession,
                tflite_session_type=sessions.TFLiteRTSession):
    # configs for each model pipeline
    cityscapes_cfg = {
        'type': settings.type,
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': datasets.CityscapesSegmentation(**settings.cityscapes_seg_calib_cfg),
        'input_dataset': datasets.CityscapesSegmentation(**settings.cityscapes_seg_val_cfg),
    }

    ade20k_cfg = {
        'type': settings.type,
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': datasets.ADE20KSegmentation(**settings.ade20k_seg_calib_cfg),
        'input_dataset': datasets.ADE20KSegmentation(**settings.ade20k_seg_val_cfg),
    }


    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_segmentation_onnx = settings.get_postproc_segmentation_onnx(save_output=settings.save_output)
    postproc_segmenation_tflite = settings.get_postproc_segmentation_tflite(save_output=settings.save_output, with_argmax=False)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        # jai-pytorch: segmentation - deeplabv3lite_mobilenetv2_tv_768x384_20190626-085932 expected_metric: 69.13% mean-iou
        'vsegcity-10300-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/jai-pytorch/deeplabv3lite_mobilenetv2_tv_768x384_20190626-085932_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # jai-pytorch: segmentation - fpnlite_aspp_mobilenetv2_tv_768x384_20200120-135701 expected_metric: 70.48% mean-iou
        'vsegcity-10301-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/jai-pytorch/fpnlite_aspp_mobilenetv2_tv_768x384_20200120-135701_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # jai-pytorch: segmentation - unetlite_aspp_mobilenetv2_tv_768x384_20200129-164340 expected_metric: 68.97% mean-iou
        'vsegcity-10302-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/jai-pytorch/unetlite_aspp_mobilenetv2_tv_768x384_20200129-164340_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # jai-pytorch: segmentation - fpnlite_aspp_regnetx800mf_768x384_20200911-144003 expected_metric: 72.01% mean-iou
        'vsegcity-10303-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/jai-pytorch/fpnlite_aspp_regnetx800mf_768x384_20200911-144003_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # jai-pytorch: segmentation - fpnlite_aspp_regnetx1.6gf_1024x512_20200914-132016 expected_metric: 75.84% mean-iou
        'vsegcity-10304-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_jai((512,1024), (512,1024), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/jai-pytorch/fpnlite_aspp_regnetx1.6gf_1024x512_20200914-132016_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # jai-pytorch: segmentation - fpnlite_aspp_regnetx3.2gf_1536x768_20200915-092738 expected_metric: 78.90% mean-iou
        'vsegcity-10305-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_jai((768,1536), (768,1536), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/jai-pytorch/fpnlite_aspp_regnetx3.2gf_1536x768_20200915-092738_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # torchvision: segmentation - torchvision deeplabv3-resnet50 - expected_metric: 73.5% MeanIoU.
        'vsegcity-30300-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_onnx((520,1040), (520,1040), backend='cv2'),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901-213517_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        # torchvision: segmentation - torchvision fcn-resnet50 - expected_metric: 71.6% MeanIoU.
        'vsegcity-30301-00':utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_onnx((520,1040), (520,1040), backend='cv2'),
            session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/cityscapes/torchvision/fcn_resnet50_1040x520_20200902-153444_opset9.onnx'),
            postprocess=postproc_segmentation_onnx
        ),
        #################################################################
        #       TFLITE MODELS
        #################mlperf models###################################
        #tensorflow-deeplab-ade20k-segmentation- deeplabv3_mnv2_ade20k_train_2018_12_03 - expected_metric: 32.04% MeanIoU.
        'vsegade-40400-00':utils.dict_update(ade20k_cfg,
            preprocess=settings.get_preproc_tflite((512, 512), (512, 512), mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
                 model_path=f'{settings.modelzoo_path}/vision/segmentation/ade20k/tf1-models/deeplabv3_mnv2_ade20k_train_2018_12_03_512x512.tflite'),
            postprocess=postproc_segmenation_tflite
        ),
    }
    return pipeline_configs

