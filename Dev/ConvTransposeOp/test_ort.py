import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('./ConvTransposeM.onnx')
x = np.ones(4 * 3 * 30 * 30).reshape((4, 3, 30, 30)).astype(np.float32)
y = session.run(None, {'X': x})[0]

np.savetxt('ort_out.txt', y.flatten(), delimiter=',', newline = ',')

