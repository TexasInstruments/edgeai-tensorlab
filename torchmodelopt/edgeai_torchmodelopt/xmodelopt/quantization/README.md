<h1> Quantization </h1>

Quantization of a deep neural 


<h2> Supported Types of Quantization </h2>

<h3> Eager Mode Quantization (v1) </h3>



<h3> FX Based Export for Quantization (v2) </h3>

The model is symbolically traced and the quantization stubs are added using the \*prepare_fx\*. The detailed documentation of the fx based quantization is available over [here](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html). We make a wrapper over the default quantization to support bias calibration, outlier suppression, among other optimizations in PTQ and QAT. Detailed documentation is available in [here](v2/README.md). Using this, quantization can be introduced in pytorch network in both Post-Training Quantization and Quatization-Aware Training Manner. 

This method could only be used if your network is completely or part-wise symbolically traceable.  




<h3> PT2E Based Export for Quantization (v3) </h3>