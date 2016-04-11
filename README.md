# mono-depth
learning depth from a single image

To run with debug statements on cpu:
THEANO_FLAGS="device=cpu,optimizer=None,compute_test_value=raise,floatX=float32" python predict_depth.py
