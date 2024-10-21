Note: 
There seems to be a confusion about the accuracy of this model. The onnx filename given by Debapriya Maji has accuracy as 49.5 / 78.0 (yolox_s_pose_ti_lite_49p5_78p0.onnx). 
However, the training log is indicating only 47.9 / 75.7.
However, when benchmarking using edgeai-benchmark, we get an accuracy that is even higher than 49.5 (https://github.com/TexasInstruments/edgeai-modelzoo/blob/master/modelartifacts/report_TDA4VM.csv)
Could it be a case of resize/pad parameters used affecting and changing the accuracy? Or is it something else?

If this difference is not resolved soon, the accuracy should be changed to yolox_s_pose_ti_lite_47p9_75p7

