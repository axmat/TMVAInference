import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

# with dynamic shape
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ['output_size'])

node = helper.make_node("Range", ['start', 'limit', 'delta'], ['Y'])

start = helper.make_value_info("start", helper.make_tensor_type_proto(1, [1]))
limit = helper.make_value_info("limit", helper.make_tensor_type_proto(1, [1]))
#delta = helper.make_value_info("delta", helper.make_tensor_type_proto(1, [1]))
delta = helper.make_tensor_value_info('delta', TensorProto.FLOAT, [1])

graph = onnx.helper.make_graph([node], "Range", [start, limit, delta], [Y])

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'RangeFloat.onnx')

# int64_t with dynamic shape
Y = helper.make_tensor_value_info('Y', TensorProto.INT64, ['output_size'])

node = helper.make_node("Range", ['start', 'limit', 'delta'], ['Y'])

start = helper.make_value_info("start", helper.make_tensor_type_proto(7, [1]))
limit = helper.make_value_info("limit", helper.make_tensor_type_proto(7, [1]))
delta = helper.make_value_info("delta", helper.make_tensor_type_proto(7, [1]))
#delta = helper.make_tensor_value_info('delta', TensorProto.FLOAT, [1])

graph = onnx.helper.make_graph([node], "Range", [start, limit, delta], [Y])

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'RangeInt.onnx')

