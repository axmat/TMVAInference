import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto, checker
import onnxruntime

start = helper.make_attribute("start", 1.)
limit = helper.make_attribute("limit", 1.)
delta = helper.make_attribute("delta", 1.)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [])

node = helper.make_node("Range", ["start", "limit", "delta"], ["Y"])
node = helper.make_node("Range", ['start', 'limit', 'delta'], ['Y'])

start = helper.make_value_info("start", helper.make_tensor_type_proto(1, [1]))
limit = helper.make_value_info("limit", helper.make_tensor_type_proto(1, [1]))
delta = helper.make_value_info("delta", helper.make_tensor_type_proto(1, [1]))

graph = onnx.helper.make_graph([node], "Range", [start, limit, delta], [Y])

model = onnx.helper.make_model(graph)
checker.check_model(model)
onnx.save(model, 'RangeFloat.onnx')

