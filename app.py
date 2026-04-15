"""
Heart Rate Monitor — Flask + Socket.IO Server
Serves the web UI and streams real-time rPPG results via WebSocket.
"""

# ── Suppress ALL warnings before any other import ──────────────────────────
import os
import warnings
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'          # suppress TF info/warning/error
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # suppress protobuf

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
# ───────────────────────────────────────────────────────────────────────────

import eventlet
eventlet.monkey_patch()   # MUST come before Flask/SocketIO imports

import sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from src.video_processor import VideoProcessor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)

socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins='*',
    max_http_buffer_size=5 * 1024 * 1024,
    ping_timeout=20,
    ping_interval=10,
)

video_processor: VideoProcessor | None = None


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------

@socketio.on('connect')
def on_connect():
    print('[WS] Client connected')
    emit('status', {'message': 'Connected to server'})


@socketio.on('disconnect')
def on_disconnect(reason=None):
    print(f'[WS] Client disconnected (reason: {reason})')
    _stop_processor()


@socketio.on('start_measurement')
def on_start(data=None):
    global video_processor
    _stop_processor()

    camera_index = int((data or {}).get('camera_index', 0))

    def on_result(result: dict):
        socketio.emit('measurement_result', result)

    video_processor = VideoProcessor(
        camera_index=camera_index,
        target_fps=30.0,
        on_result=on_result,
        jpeg_quality=65,        # lower = smaller payload, less WebSocket lag
    )
    success = video_processor.start()

    if success:
        emit('status', {'message': 'Camera started', 'camera_index': camera_index})
    else:
        video_processor = None
        emit('error', {'message': f'Could not open camera {camera_index}. '
                                  'Check permissions or try a different index.'})


@socketio.on('stop_measurement')
def on_stop():
    _stop_processor()
    emit('status', {'message': 'Measurement stopped'})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stop_processor():
    global video_processor
    if video_processor and video_processor.is_running:
        video_processor.stop()
    video_processor = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    host  = os.getenv('HOST',  '0.0.0.0')
    port  = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    print(f"\n{'='*55}")
    print(f"  Heart Rate Monitor")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*55}\n")

    socketio.run(app, host=host, port=port, debug=debug)