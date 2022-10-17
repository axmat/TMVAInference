import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [5])

node = helper.make_node(
    'Sign',
    ['X'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Sign',
    [X],
    [Y]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'Sign.onnx')

