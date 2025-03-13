import base64
import io
import numpy as np
import onnxruntime as ort
import socketio
import eventlet
from PIL import Image
import torchvision.transforms as transforms
import torch  # Only used here for tensor stacking; ONNX Runtime requires numpy arrays

# ==========================
# 1. Set Up SocketIO Server
# ==========================
sio = socketio.Server()
app = socketio.WSGIApp(sio)

# ==========================
# 2. Load ONNX Model
# ==========================
# Create an inference session for your ONNX model.
ort_session = ort.InferenceSession("model.onnx")
# Print input name for debugging.
input_name = ort_session.get_inputs()[0].name
print("ONNX model input name:", input_name)

# ==========================
# 3. Define Preprocessing Transform
# ==========================
# This transform should match what you used during training.
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==========================
# 4. Set Up Image Buffer for Temporal Sequences
# ==========================
sequence_length = 5  # Number of consecutive frames required
image_buffer = []    # Will store frames of shape (num_views, channels, height, width)

# ==========================
# 5. SocketIO Event Handlers
# ==========================
@sio.on("telemetry")
def telemetry(sid, data):
    """
    This event is triggered whenever the simulator sends telemetry.
    It decodes the incoming image, applies preprocessing, and adds it to the image buffer.
    Once enough frames are collected, the model is run for inference.
    """
    if data is None:
        return

    # Decode the base64 image sent from the simulator.
    img_string = data["image"]
    image = Image.open(io.BytesIO(base64.b64decode(img_string)))
    
    # Apply preprocessing transform.
    img_tensor = transform(image)  # Shape: (3, 66, 200)
    
    # Simulate three camera views by replicating the center image.
    # This produces a tensor of shape: (num_views=3, channels=3, height=66, width=200)
    view_tensor = torch.stack([img_tensor, img_tensor, img_tensor], dim=0)
    
    # Convert the tensor to a numpy array (for ONNX Runtime).
    frame = view_tensor.numpy()  # Shape: (3, 3, 66, 200)
    
    # Append the new frame to the image buffer.
    image_buffer.append(frame)
    if len(image_buffer) > sequence_length:
        image_buffer.pop(0)
    
    # If we don't yet have enough frames, do not run inference.
    if len(image_buffer) < sequence_length:
        return
    
    # Construct the model input.
    # Expected shape: (batch, sequence_length, num_views, channels, height, width)
    sequence_input = np.stack(image_buffer, axis=0)  # (5, 3, 3, 66, 200)
    sequence_input = np.expand_dims(sequence_input, axis=0)  # (1, 5, 3, 3, 66, 200)
    
    # Run inference with ONNX Runtime.
    outputs = ort_session.run(None, {input_name: sequence_input})
    # Assume outputs[0] contains the predictions (shape: (1, 3)).
    prediction = outputs[0][0]  # Get the first (and only) batch element.
    steering, throttle, brake = prediction  # Unpack control commands.
    
    print(f"Predicted controls: steering={steering:.3f}, throttle={throttle:.3f}, brake={brake:.3f}")
    
    # Send control commands back to the simulator.
    send_control(steering, throttle, brake)

@sio.on("connect")
def connect(sid, environ):
    print("Client connected:", sid)

def send_control(steering, throttle, brake):
    """
    Sends control commands back to the simulator.
    """
    data = {
        "steering_angle": str(steering),
        "throttle": str(throttle),
        "brake": str(brake)
    }
    sio.emit("steer", data=data)

# ==========================
# 6. Start the Server
# ==========================
if __name__ == "__main__":
    print("Starting SocketIO server on port 4567")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
