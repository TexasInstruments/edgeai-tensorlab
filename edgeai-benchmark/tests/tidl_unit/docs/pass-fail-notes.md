## Pass/Fail Notes

- Compilation:
    - If TIDL compilation exits gracefully with exit code of 0, it is defined as PASS.
    - If TIDL compilation fails or times out (given by --timeout), it is defined as FAIL

- Inference:
    - If TIDL inferecnce exits gracefully with exit code of 0 and normalized mean-squared-error (NMSE) with golden output is less than ``threshold`` (given in tidl_unit.yaml), it is defined as PASS.
    - If TIDL inference fails, times out or exceeds NMSE ``threshold``, it is defined as FAIL

