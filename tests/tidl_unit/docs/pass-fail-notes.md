## Pass/Fail Notes

- Test are defined as failing in pytest if they throw an exception. If compilation fails but exits gracefully, it will be defined as a pass by PyTest

- Inference will report a fail if the normalized mean-squared-error (NMSE) is above a threshold
    - If there are multiple outputs in a test, the maximum normalized mean-squared-error is selected
    - The threshold is defined in tidl_unit.yaml under the key "inference_nmse_thresholds". A default threshold is defined and may be overriden on a per-test level



