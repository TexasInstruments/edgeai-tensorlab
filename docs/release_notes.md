
## Release 8.1 (08_01_00_05)
- Date: 21 December 2021
- Software manifest: https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00_05/edgeai-modelzoo_08_01_manifest.html
- Model artifacts list: https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00_05/modelartifacts/8bits/artifacts.csv
- Accuracy report: https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00_05/modelartifacts/report_20211215-133050.csv (Note: Object detection models are run with thresholds tuned for performance and not for accuracy - so ignore the accuracies reported for them)
- This repository has been restructured. We now use the original mmdetection and added our changes on top of it as commits. This will help us to rebase from mmdetection as more and more features are added there. Please see more details in the usage.
- We now have a tool that can automatically replace some of embedded un-friendly layers. Please see more details in the usage.
- Note: YOLOv5 models are not part of this repository but providing the link here: https://github.com/TexasInstruments/edgeai-yolov5/
- Note: YOLOv5 model artifacts are not part of this repository but providing the links here:  https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_01_00_05/edgeai-yolov5/pretrained_models/modelartifacts/8bits/artifacts.csv
