import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# Test 3: 3D
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 2, 2, 2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 3, 3, 3])

node = helper.make_node(
    'ConvTranspose',
    ['X', 'W'],
    ['Y'],
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 2, 2, 2, 2],
    [1.] * 16
)

graph = onnx.helper.make_graph(
    [node],
    'ConvTranspose3d',
    [X, W],
    [Y],
    [tensor_W]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'ConvTranspose3d.onnx')


