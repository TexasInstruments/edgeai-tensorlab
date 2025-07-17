# ONNX Backend Tests

[ONNX backend tests](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md) are a suite of tests written by the ONNX community. It is made up of a large number of single-operator ONNX models with inputs and expected outputs. It not only tests an entire ONNX opset, but also tests each operator under a variety of input parameters. 

## Setup

To use these tests, ensure that you have installed pytest as well as helpful plugins: `pip install pytest pytest-xdist pytest-html==3.2.0`

## Documentation
Usage notes: [usage-notes.md](docs/usage-notes.md)

Pass/Fail Notes: [pass-fail-notes.md](docs/pass-fail-notes.md)

Code Outline: [code-outline.md](docs/code-outline.md)



