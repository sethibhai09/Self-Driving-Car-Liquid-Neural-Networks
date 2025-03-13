import socketio
import eventlet
import eventlet.wsgi
import logging
from flask import Flask

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Socket.IO server with CORS allowed.
sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
# Wrap the Flask app with Socket.IO.
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

@sio.on('connect')
def connect(sid, environ):
    logger.info("Client connected: %s", sid)

@sio.on('telemetry')
def telemetry(sid, data):
    # Log receipt of telemetry.
    logger.info("Telemetry event received from %s", sid)
    logger.info("Telemetry data: %s", data)
    # For testing, send fixed control commands (try a strong throttle).
    fixed_data = {
        "steering_angle": 0.0,  # No steering correction.
        "throttle": 0.5,        # Maximum throttle.
        "brake": 0.0
    }
    logger.info("Sending fixed control: %s", fixed_data)
    sio.emit("steer", fixed_data)

@sio.on('disconnect')
def disconnect(sid):
    logger.info("Client disconnected: %s", sid)

if __name__ == '__main__':
    logger.info("Starting Socket.IO server on port 4567")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
