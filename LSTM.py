import torch
from torch import nn
from torch.nn import RNN
import onnx
import onnxruntime
import numpy as np

torch.set_default_dtype(torch.float32)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = LSTM(2, 4, 1, batch_first=False, bidirectional=True)

    def forward(self, input, h0, c0):
        out, _, _ = self.net(input, h0, c0)
        return out

model = Model()

# set the weights
model.rnn.weight_ih_l0.data = torch.FloatTensor(
[[1.16308, 2.21221],
 [0.483805, 0.774004],
 [0.299563, 1.04344],
 [0.153025, 1.18393]])
model.rnn.weight_ih_l0_reverse.data = torch.FloatTensor(
[[-1.16881, 1.89171],
 [1.55807, -1.23474],
 [-0.545945, -1.77103],
 [-2.35563, -0.451384]])
# set recurrence
model.rnn.weight_hh_l0.data = torch.FloatTensor(
[[-0.264848, -1.30311, 0.0712087, 0.64198],
 [-2.76538, -0.652074, -0.784275, -1.76749],
 [-0.450673, -0.917929, -0.966654, 0.650856],
 [0.285538, -0.909848, -1.90459, -0.140926]])
model.rnn.weight_hh_l0_reverse.data = torch.FloatTensor(
[[-1.37131, 0.780644, 0.441009, 1.15856],
 [0.313298, 1.96766, -1.11991, -0.00440959],
 [0.407622, 2.60569, -0.840986, 0.585658],
 [0.823292, -0.696818, 1.15115, 0.150269]])
# set the bias
model.rnn.bias_ih_l0.data = torch.FloatTensor(
[-0.161029, -2.58991, 0.339721, -0.31664])
model.rnn.bias_hh_l0.data = torch.FloatTensor(
[0.049053, -1.89795, -0.327121, -0.159628])
model.rnn.bias_ih_l0_reverse.data = torch.FloatTensor(
[-0.183054, -0.977459, -1.08309, -0.0165881])
model.rnn.bias_hh_l0_reverse.data = torch.FloatTensor(
[1.99349, 1.35513, -0.697978, -0.708618])
# set initial h
h0 = torch.FloatTensor(
[-0.371075, 0.252533, -1.42195, 0.39303,
 -0.463112, -1.02438, -0.538399, -2.21508,
 -1.4221, -0.149365, 1.2587, 1.38294,
 -0.0841612, 1.45697, 0.0679387, 2.11548,
 -1.51051, 1.50948, 0.206351, -0.981445,
 -0.221477, -0.230484, 0.453313, 0.795476]).reshape(2, 3, 4)

#for name, param in model.named_parameters():
#    print(name, param)
model.eval()
input = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) / 100.
#print(x)
#x.fill_(.1)
output = model(input, h0)
#print(output.shape)
#print(output)
#y = output.reshape(18, 4)
#print(y)
#out = y.reshape(3, 2, 3, 4)
#print("\n\nout\n")
#print(out)
#print("\nlast")
#print(y_h)

#y = model(x)
torch.onnx.export(
    model,
    (input, h0),
    'RNN_bidirectional.onnx',
    export_params = True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input', 'h0'],
    output_names=['output']
)

# Check model
onnx_model = onnx.load("RNN_bidirectional.onnx")
onnx.checker.check_model(onnx_model)

# Validate
#ort_session = onnxruntime.InferenceSession("RNN_bidirectional.onnx")
#def to_numpy(tensor):
#    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
                # ort_session.get_inputs()[1].name: to_numpy(h0)}
#ort_outs = ort_session.run(None, ort_inputs)

#np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
print("SUCCES")

#y = torch.from_numpy(ort_outs[0])
#y = y.reshape(3, 3, 2, 4)
#x = y.permute(0, 2, 1, 3)
#x = y.view(3, 2, 3, 4)
#print(x.shape)
#print(x.reshape(18, 4))

#print("yh")
#print(ort_outs[1])

#print(y.view(3, 2, 3, 4))
#print(y.flatten().reshape(3, 3, 2, 4).reshape(3, 2, 3, 4).reshape(18, 4))
#y = ort_outs[0].reshape(3, 3, 2, 4)
#print(y.reshape(18, 4))
