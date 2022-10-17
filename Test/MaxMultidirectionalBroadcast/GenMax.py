import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [3, 1])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3, 1])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])
# Y shape should be [2, 3, 4]

#tensor_A = helper.make_tensor('A', TensorProto.FLOAT, [3, 1],
#    [0.35974154, -2.20873388,  0.95746274])
#tensor_B = helper.make_tensor('B', TensorProto.FLOAT, [2, 3, 1],
#    [0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695])
#tensor_C = helper.make_tensor('C', TensorProto.FLOAT, [1, 4],
#    [-0.41468981, -0.46591926,  0.56172534,  0.05616931])

node = helper.make_node('Max', ['A', 'B', 'C'], ['Y'])

graph = onnx.helper.make_graph([node], 'Max', [A, B, C], [Y])
#, [tensor_A, tensor_B, tensor_C])

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'MaxMultidirectionalBroadcast.onnx')

