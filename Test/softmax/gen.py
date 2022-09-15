import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# 1d
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3])

node = helper.make_node(
    'Softmax',
    ['X'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Softmax',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Softmax1d.onnx')

# 2d
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])

node = helper.make_node(
    'Softmax',
    ['X'],
    ['Y'],
    axis = 1
)

graph = onnx.helper.make_graph(
    [node],
    'Softmax',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Softmax2d.onnx')

# 3d
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4])

node = helper.make_node(
    'Softmax',
    ['X'],
    ['Y'],
    axis = 1
)

graph = onnx.helper.make_graph(
    [node],
    'Softmax',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Softmax3d.onnx')

# 4d
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4, 2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4, 2])

node = helper.make_node(
    'Softmax',
    ['X'],
    ['Y'],
    axis = -2
)

graph = onnx.helper.make_graph(
    [node],
    'Softmax',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Softmax4d.onnx')


