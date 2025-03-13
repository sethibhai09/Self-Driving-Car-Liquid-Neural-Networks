import numpy as np
import onnxruntime as ort

# Create dummy input with the correct shape.
dummy_input = np.random.randn(1, 5, 3,3, 66, 200).astype(np.float32)

# Create an inference session.
ort_session = ort.InferenceSession("model.onnx")

# Use the correct input name from the model.
outputs = ort_session.run(None, {"onnx::Reshape_0": dummy_input})
print(outputs)
