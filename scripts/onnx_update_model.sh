# to apply this script - run this from the top level folder:
# find ./models -name "*.onnx" -exec ./scripts/onnx_update_models.sh {} \;

echo updating $1
python3 ../../processor-sdk-vision/psdkra_latest/onnxruntime/tools/python/remove_initializer_from_input.py --input $1 --output $1
python3 ./scripts/onnx_shape_inference.py --input $1 --output $1
