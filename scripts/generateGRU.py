import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker

# Test 1: gru defaults
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 15, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 15, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 5])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=3,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 15, 2],
    [0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 15, 5],
    [0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'GRUDefaults',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'GRUDefaults.onnx')

# Test 2: gru initial_bias
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 9, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 9, 3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 18])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 3])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 3])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R', 'B'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=3,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 9, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 9, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_B = helper.make_tensor(
    'B',
    TensorProto.FLOAT,
    [1, 18],
    [1., 1., 1., 1., 1., 1., 1., 1., 1.,
     0., 0., 0., 0., 0., 0., 0., 0., 0.]
)

graph = onnx.helper.make_graph(
    [node],
    'GRUInitialBias',
    [X, W, R, B],
    [Y, Y_h],
    [tensor_W, tensor_R, tensor_B]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'GRUInitialBias.onnx')

# Test 3: gru seq_length
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 15, 3])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 15, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 1, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [1, 3, 5])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=5,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 15, 3],
    [0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 15, 5],
    [0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'GRUSeqLength',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'GRUSeqLength.onnx')

# Test 4: gru batchwise
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 1, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 18, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 18, 6])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 1, 6])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [3, 1, 6])

node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='forward',
    hidden_size=6,
    layout=1
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [1, 18, 2],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [1, 18, 6],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
)

graph = onnx.helper.make_graph(
    [node],
    'GRUBatchwise',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'GRUBatchwise.onnx')

# Test 5: gru bidirectional
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [2, 15, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [2, 15, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 3, 5])
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, [2, 3, 5])
node = helper.make_node(
    'GRU',
    ['X', 'W', 'R'],
    ['Y', 'Y_h'],
    "",
    activations=['Sigmoid', 'Tanh', 'Tanh', 'Sigmoid', 'Tanh', 'Tanh'],
    clip=0.,
    direction='bidirectional',
    hidden_size=5,
    layout=0
)

tensor_W = helper.make_tensor(
    'W',
    TensorProto.FLOAT,
    [2, 15, 2],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
)
tensor_R = helper.make_tensor(
    'R',
    TensorProto.FLOAT,
    [2, 15, 5],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)

graph = onnx.helper.make_graph(
    [node],
    'GRUBidirectional',
    [X, W, R],
    [Y, Y_h],
    [tensor_W, tensor_R]
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'GRUBidirectional.onnx')

