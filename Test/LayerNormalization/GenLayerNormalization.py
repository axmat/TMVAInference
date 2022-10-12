import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# 2d
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
onnx.save(model, 'LayerNormalization2d.onnx')

# 3d
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4, 5])
Scale = helper.make_tensor_value_info('Scale', TensorProto.FLOAT, [4, 5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4, 5])

node = helper.make_node(
    'LayerNormalization',
    ['X', 'Scale', 'B'],
    ['Y'],
    axis=2
)

tensor_Scale = helper.make_tensor('Scale',
    TensorProto.FLOAT,
    [4, 5],
    [0.1] * 20,
)

tensor_B = helper.make_tensor('B',
    TensorProto.FLOAT,
    [4, 5],
    [0.2] * 20
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
onnx.save(model, 'LayerNormalization3d.onnx')

