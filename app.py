from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

import cv2
from flask import Response, stream_with_context

from vision import process_frame

app = Flask(__name__)
CORS(app)  # Allow access from browser

# Simulated robot state
robot_state = {
    "fsm_state": "idle",
    "dx": 0.0,
    "shingle_count": 0
}

# Add a global VideoCapture object
camera = cv2.VideoCapture(0)  # Or 1, depending on your webcam

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, result = process_frame(frame)

            # You could store or forward `result["dx"]` to robot FSM here

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/status")
def status():
    return jsonify(robot_state)

@app.route("/command", methods=["POST"])
def command():
    data = request.get_json()
    cmd = data.get("cmd")

    if cmd == "start":
        robot_state["fsm_state"] = "aligning"
    elif cmd == "place_shingle":
        robot_state["shingle_count"] += 1
        robot_state["fsm_state"] = "placing"
    elif cmd == "stop":
        robot_state["fsm_state"] = "stopped"
    else:
        return jsonify({"error": "Unknown command"}), 400

    return jsonify({"success": True, "new_state": robot_state})

# ✅ This must be above the app.run() line!
@app.route("/")
def dashboard():
    return send_from_directory("static", "index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

