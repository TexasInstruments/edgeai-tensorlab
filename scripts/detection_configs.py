# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Usage:
(1) Use one of the following config files.
(2) Inside the config file, make sure that the dataset that needs to be trained on is uncommented.
(3) Use the appropriate input resolution in the config file (input_size).
(4) Recommend to run the first training with voc0712 dataset as it is widely used and reasonably small.
(5) To convert cityscapes to coco format, run the script: tools/convert_datasets/cityscapes.py

config='./configs/edgeailite/ssd/ssd_regnetx_fpn_bgr_lite.py'
config='./configs/edgeailite/ssd/ssd_mobilenet_fpn_lite.py'
config='./configs/edgeailite/ssd/ssd_mobilenet_lite.py'
config='./configs/edgeailite/ssd/ssd_resnet_fpn.py'

config='./configs/edgeailite/retinanet/retinanet_regnet_fpn_bgr_lite.py'
config='./configs/edgeailite/retinanet/retinanet_resnet_fpn.py'

config='./configs/edgeailite/yolov3/yolov3_regnet_bgr_lite.py'
config='./configs/edgeailite/yolov3/yolov3_d53_relu.py'
config='./configs/edgeailite/yolov3/yolov3_d53.py'

config='./configs/edgeailite/yolox/yolox_nano_lite.py'
config='./configs/edgeailite/yolox/yolox_tiny_lite.py'
config='./configs/edgeailite/yolox/yolox_s_lite.py'
config='./configs/edgeailite/yolox/yolox_m_lite.py'

# tiny models
config='./configs/edgeailite/ssd/ssd_mobilenetp5_lite_320x320.py'
config='./configs/edgeailite/yolox/yolox_pico_lite.py'
config='./configs/edgeailite/yolox/yolox_femto_lite.py'
'''

config='./configs/edgeailite/yolox/yolox_nano_lite.py'
