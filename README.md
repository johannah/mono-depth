# mono-depth
learning depth from a single image

# install
conda env create -f mono_environment.yml

To run with debug statements on cpu:
THEANO_FLAGS="device=cpu,optimizer=None,compute_test_value=raise,floatX=float32" python predict_depth.py

To run on cpu:
THEANO_FLAGS="device=cpu,floatX=float32" python predict_depth.py

To run on gpu with no debug:
THEANO_FLAGS="device=gpu,floatX=float32" python predict_depth.py
