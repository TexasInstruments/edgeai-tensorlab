## Latest updates

* [2022-January]: ONNX+Prototxt export of YOLOX models have been fixed. The folder configs/edgeailite/yolox now has config files for our lite versions of YOLOX models. YOLOX models are highly optimal for embedded devices interms of accuracy and FPS.

* [2021-December]: This repository is now a fork of [mmdetection](https://github.com/open-mmlab/mmdetection). This will enable us to rebase and take changes from mmdetection easily.

* [2021-Auguest]: SSD, RetinaNet, YOLOv3 configs are now enabled under configs/edgeailite. These configs enable training of embedded friendly lite models. ONNX+Prototxt export is supported for these models (required for TIDL). A config file can be selected via scripts/detection_configs.py before running ./run_detection_train.sh, ./run_detection_test.sh or ./run_detection_export.sh 
