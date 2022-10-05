import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# 1: {5} to {4, 5}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4, 5])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast1.onnx')

# 2: {5} to {2, 3, 4, 5}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3, 4, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4, 5])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast2.onnx')

# 3: {2, 1, 1, 5} to {2, 3, 4, 5}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 1, 1, 5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3, 4, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4, 5])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast3.onnx')

# 4: {2, 1} to {2, 4}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 1])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast4.onnx')

# 5: {2, 1, 4} to {2, 3, 4}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 1, 4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast5.onnx')

# 6: {2, 1, 3, 1, 2} to {2, 2, 3, 2, 2}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 1, 3, 1, 2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 2, 3, 2, 2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 2, 3, 2, 2])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast6.onnx')


# 7: Broadcast boths {2, 1, 3, 1} {1, 1, 3, 4} to {2, 1, 3, 4}
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 1, 3, 1])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 1, 3, 4])

node = helper.make_node(
    'Add',
    ['A', 'B'],
    ['Y'],
)

graph = onnx.helper.make_graph(
    [node],
    'Add',
    [A, B],
    [Y],
)

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'AddBroadcast7.onnx')

