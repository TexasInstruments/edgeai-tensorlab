## Release 8.6 (08_06_00_01)
- Date: 1 March 2023
- Release tag: r8.6
- [Models & Documentation](https://github.com/TexasInstruments/edgeai-modelzoo/tree/r8.6)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/manifest.html)
- [Compiled Model artifacts](../modelartifacts)
- [Compiled Model artifacts list for TDA4VM](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/modelartifacts/TDA4VM/8bits/artifacts.csv)
- [Compiled Model artifacts list for AM62A](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/modelartifacts/AM62A/8bits/artifacts.csv)
- [Compiled Model artifacts list for AM68A](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/modelartifacts/AM68A/8bits/artifacts.csv)
- [Compiled Model artifacts list for AM69A](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/modelartifacts/AM69A/8bits/artifacts.csv)


## Release 8.2 (08_02_00_11)
- Date: 6 April 2022
- Release tag: r8.2
- [Models Code & Documentation](https://github.com/TexasInstruments/edgeai-modelzoo/tree/r8.2)
- [Compiled Model artifacts list at ti.com](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_02_00_11/modelartifacts/8bits/artifacts.csv)
- [Release Manifest](https://software-dl.ti.com/jacinto7/esd/modelzoo/08_02_00_11/manifest.html)

#### Other models
- [Models Code & Documentation edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5/tree/r8.2)
- [Compiled Model artifacts list for edgeai-yolov5](https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/modelartifacts/8bits/artifacts.csv)
- [Release Manifest for edgeai-yolov5](https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/manifest.html)


## Release 8.1 (08_01_00_05)
- Date: 21 December 2021
- Software manifest: https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00_05/edgeai-modelzoo_08_01_manifest.html
- Model artifacts list: https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00_05/modelartifacts/8bits/artifacts.csv
- Accuracy report: https://software-dl.ti.com/jacinto7/esd/modelzoo/08_01_00_05/modelartifacts/report_20211215-133050.csv (Note: Object detection models are run with thresholds tuned for performance and not for accuracy - so ignore the accuracies reported for them)
- This repository has been restructured. We now use the original mmdetection and added our changes on top of it as commits. This will help us to rebase from mmdetection as more and more features are added there. Please see more details in the usage.
- We now have a tool that can automatically replace some of embedded un-friendly layers. Please see more details in the usage of [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision).
- Note: YOLOv5 models are not part of this repository but providing the link here: https://github.com/TexasInstruments/edgeai-yolov5/
- Note: YOLOv5 model artifacts are not part of this repository but providing the links here:  https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_01_00_05/edgeai-yolov5/pretrained_models/modelartifacts/8bits/artifacts.csv
