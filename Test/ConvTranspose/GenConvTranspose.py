import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# Test 2: 1D
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 5])

node = helper.make_node(
    'ConvTranspose',
    ['X', 'W'],
    ['Y'],
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 2, 3],
    [1.] * 6
)

graph = onnx.helper.make_graph(
    [node],
    'ConvTranspose1d',
    [X, W],
    [Y],
    [tensor_W]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'ConvTranspose1d.onnx')

# Test 1: 2D
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 3, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 5, 5])

node = helper.make_node(
    'ConvTranspose',
    ['X', 'W'],
    ['Y'],
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 2, 3, 3],
    [1.] * 18
)

graph = onnx.helper.make_graph(
    [node],
    'ConvTranspose2d',
    [X, W],
    [Y],
    [tensor_W]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'ConvTranspose2d.onnx')

# Test 3: 3D
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3, 4, 5])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 3, 3, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 5, 6, 7])

node = helper.make_node(
    'ConvTranspose',
    ['X', 'W'],
    ['Y'],
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 2, 3, 3, 3],
    [1.] * 54
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

# 2D with bias
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 3, 3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 5, 5])

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
onnx.save(model, 'ConvTransposeBias2d.onnx')

# Grouped 2D
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [2, 2, 3, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4, 5, 5])

node = helper.make_node(
    'ConvTranspose',
    ['X', 'W'],
    ['Y'],
    group = 2
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [2, 2, 3, 3],
    [1.] * 36
)

graph = onnx.helper.make_graph(
    [node],
    'ConvTranspose2d',
    [X, W],
    [Y],
    [tensor_W]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'ConvTransposeGrouped2d.onnx')

