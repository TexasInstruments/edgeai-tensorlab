
## Guidelines for Model training & quantization
Quantization (especially 8-bit Quantization) is important to get best throughput for inference. Quantization can be done using either Post Training Quantization (PTQ) or Quantization Aware Training (QAT). Guidelines for Model training and tools for QAT are given the **[documentation on Quantization](./quantization/docs/Quantization.md)**.

- Post Training Quantization (PTQ): TIDL natively supports PTQ - it can take floating point models and can quantize them using advanced calibration methods. In the above link, we have provided guidelines on how to choose models and how to train them for best accuracy with quantization - these guidelines are important to reduce accuracy drop during quantization with PTQ. 

- Quantization Aware Training (QAT): In spite of following these guidelines, if there are models that have significant accuracy drop with PTQ, it is possible to improve the accuracy using QAT. See the above link for more details.

