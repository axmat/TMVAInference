import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 4])
Scale = helper.make_tensor_value_info('Scale', TensorProto.FLOAT, [4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

node = helper.make_node(
    'LayerNormalization',
    ['X', 'Scale', 'B'],
    ['Y'],
)

tensor_Scale = helper.make_tensor('Scale',
    TensorProto.FLOAT,
    [4],
    [0.5, -0.2, 0.3, 1.],
)

tensor_B = helper.make_tensor('B',
    TensorProto.FLOAT,
    [4],
    [0.2, -0.1, 0.1, 0.],
)

graph = onnx.helper.make_graph(
    [node],
    'LayerNormalization',
    [X, Scale, B],
    [Y],
    [tensor_Scale, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'LayerNormalization.onnx')

