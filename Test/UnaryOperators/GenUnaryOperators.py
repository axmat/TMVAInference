import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])

node = helper.make_node(
    'Sqrt',
    ['X'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Sqrt',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Sqrt.onnx')


X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])

node = helper.make_node(
    'Reciprocal',
    ['X'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Reciprocal',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Reciprocal.onnx')

# Exp
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [10])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [10])

node = helper.make_node(
    'Exp',
    ['X'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Exp',
    [X],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Exp.onnx')


