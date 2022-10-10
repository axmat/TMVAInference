import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# 2D with bias
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 1, 5, 5])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 3, 3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 1, 3, 3])

node = helper.make_node(
    'ConvTranspose',
    ['X', 'W', 'B'],
    ['Y'],
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 2, 3, 3],
    [1.] * 18
)

tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [2],
    [1., 2.]
)

graph = onnx.helper.make_graph(
    [node],
    'ConvTranspose2d',
    [X, W, B],
    [Y],
    [tensor_W, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'ConvTransposeBiasBatched.onnx')

