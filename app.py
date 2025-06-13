from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import sys
import threading
import time

import cv2
from flask import Response, stream_with_context

from vision import (
    process_frame, calibrate_vision_system, get_vision_parameters, set_vision_parameters,
    get_vision_presets, save_vision_preset, load_vision_preset, delete_vision_preset
)
from nucleo_comms import NucleoComms

# Import simulation components
from simulation import (
    initialize_simulation, get_simulated_nucleo, get_simulated_vision, 
    get_simulation_config, update_simulation_config, is_simulation_mode,
    process_simulated_frame
)

app = Flask(__name__)
CORS(app)  # Allow access from browser

# Robot state with motor integration and FSM parameters
robot_state = {
    "fsm_state": "idle",
    "dx": 0.0,
    "shingle_count": 0,
    "motor_status": {
        "connected": False,
        "x_position": 0.0,
        "y_position": 0.0,
        "z_position": 0.0,
        "x_running": False,
        "y_running": False,
        "z_running": False
    },
    # FSM control parameters
    "fsm_params": {
        "alignment_tolerance": 0.005,      # meters - how close to target before considering aligned
        "scan_duration": 2.0,              # seconds - how long to scan for shingle edges
        "placement_depth": 0.02,           # meters - how far to lower for shingle placement
        "nail_duration": 1.0,              # seconds - how long to activate nail gun
        "advance_distance": 0.15,          # meters - distance to advance for next shingle
        "max_state_timeout": 30.0,         # seconds - max time in any state before error
        "vision_update_rate": 10.0,        # Hz - how often to process vision data
        "manual_override": False           # flag to disable automatic FSM progression
    },
    # FSM timing and status
    "fsm_status": {
        "state_start_time": time.time(),
        "last_vision_update": time.time(),
        "error_message": "",
        "scan_complete": False,
        "alignment_complete": False,
        "placement_complete": False,
        "nail_complete": False
    }
}

# Check for simulation mode
SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'false').lower() == 'true' or '--simulate' in sys.argv

# Initialize hardware connections
if SIMULATION_MODE:
    print("🤖 Starting in SIMULATION MODE")
    initialize_simulation()
    camera = None  # No real camera in simulation
    nucleo = get_simulated_nucleo()
else:
    print("🔧 Starting in HARDWARE MODE")
    camera = cv2.VideoCapture(0)  # Or 1, depending on your webcam
    nucleo = NucleoComms()  # STM32 Nucleo communication

# Background thread for motor status polling
def motor_status_thread():
    """Background thread to continuously poll motor status from STM32"""
    while True:
        try:
            if nucleo.is_connected():
                status = nucleo.read_motor_status()
                if status:
                    robot_state["motor_status"].update({
                        "connected": True,
                        "x_position": status.get("x_position", 0.0),
                        "y_position": status.get("y_position", 0.0),
                        "z_position": status.get("z_position", 0.0),
                        "x_running": status.get("x_running", False),
                        "y_running": status.get("y_running", False),
                        "z_running": status.get("z_running", False)
                    })
                else:
                    robot_state["motor_status"]["connected"] = False
            else:
                robot_state["motor_status"]["connected"] = False
                # Try to reconnect
                if not nucleo.connect():
                    time.sleep(5)  # Wait before retry
                    
        except Exception as e:
            print(f"Motor status thread error: {e}")
            robot_state["motor_status"]["connected"] = False
        
        time.sleep(0.1)  # Poll at 10Hz

# Start motor status monitoring thread
status_thread = threading.Thread(target=motor_status_thread, daemon=True)
status_thread.start()


# =============================================================================
# FINITE STATE MACHINE IMPLEMENTATION
# =============================================================================

def transition_to_state(new_state: str, reset_flags: bool = True):
    """
    Safely transition to a new FSM state with proper logging and timing
    
    Args:
        new_state: Target state to transition to
        reset_flags: Whether to reset completion flags (default True)
    """
    old_state = robot_state["fsm_state"]
    robot_state["fsm_state"] = new_state
    robot_state["fsm_status"]["state_start_time"] = time.time()
    robot_state["fsm_status"]["error_message"] = ""
    
    if reset_flags:
        robot_state["fsm_status"]["scan_complete"] = False
        robot_state["fsm_status"]["alignment_complete"] = False
        robot_state["fsm_status"]["placement_complete"] = False
        robot_state["fsm_status"]["nail_complete"] = False
    
    print(f"FSM: {old_state} → {new_state}")


def check_state_timeout() -> bool:
    """
    Check if current state has exceeded maximum timeout
    
    Returns:
        bool: True if state has timed out
    """
    elapsed = time.time() - robot_state["fsm_status"]["state_start_time"]
    max_timeout = robot_state["fsm_params"]["max_state_timeout"]
    
    if elapsed > max_timeout:
        robot_state["fsm_status"]["error_message"] = f"State timeout after {elapsed:.1f}s"
        return True
    return False


def is_motor_motion_complete() -> bool:
    """
    Check if all motors have completed their current motion
    
    Returns:
        bool: True if no motors are currently running
    """
    motor_status = robot_state["motor_status"]
    return not (motor_status["x_running"] or motor_status["y_running"] or motor_status["z_running"])


def fsm_idle_state():
    """
    IDLE STATE: Robot is waiting for commands
    
    Transitions:
        - Manual command "start" → scanning
        - Manual command "home" → homing
        - Manual command "emergency_stop" → emergency_stop
        - Manual move commands → manual_mode
    
    This is the default state where the robot waits for operator input.
    All systems are ready but no autonomous operation is occurring.
    """
    # Idle state is purely reactive - transitions happen via manual commands
    pass


def fsm_scanning_state():
    """
    SCANNING STATE: Robot scans for shingle edges to determine placement position
    
    Transitions:
        - Scan duration completed + valid edge detected → aligning
        - Scan duration completed + no edge detected → error
        - Timeout or error → error
    
    The robot uses its downward-facing camera to detect existing shingle edges.
    Vision processing calculates dx (lateral offset) needed for proper alignment.
    """
    elapsed = time.time() - robot_state["fsm_status"]["state_start_time"]
    scan_duration = robot_state["fsm_params"]["scan_duration"]
    
    # Update vision data continuously during scanning
    current_time = time.time()
    vision_rate = robot_state["fsm_params"]["vision_update_rate"]
    if current_time - robot_state["fsm_status"]["last_vision_update"] > (1.0 / vision_rate):
        robot_state["fsm_status"]["last_vision_update"] = current_time
        # Vision data is updated in gen_frames() - dx value is available in robot_state["dx"]
    
    # Check if scanning duration is complete
    if elapsed >= scan_duration:
        robot_state["fsm_status"]["scan_complete"] = True
        
        # Validate that we have meaningful vision data
        dx = robot_state["dx"]
        if abs(dx) < 1.0:  # Reasonable dx value (less than 1 meter offset)
            print(f"FSM: Scan complete, detected dx = {dx:.3f}m")
            transition_to_state("aligning")
        else:
            robot_state["fsm_status"]["error_message"] = f"Invalid scan result: dx = {dx:.3f}m"
            transition_to_state("error")


def fsm_aligning_state():
    """
    ALIGNING STATE: Robot moves laterally to align with detected shingle edge
    
    Transitions:
        - Motor motion complete + within tolerance → placing
        - Motor motion complete + outside tolerance → aligning (retry)
        - Motor error or timeout → error
    
    Uses dx from vision system to calculate lateral movement needed.
    Moves X-axis to position robot over the correct shingle placement location.
    """
    dx = robot_state["dx"]
    tolerance = robot_state["fsm_params"]["alignment_tolerance"]
    
    # Check if we need to start alignment motion
    if not robot_state["fsm_status"]["alignment_complete"]:
        if abs(dx) > tolerance:
            print(f"FSM: Starting alignment move, dx = {dx:.3f}m")
            success = nucleo.send_position_command(x_dist=dx)
            if success:
                robot_state["fsm_status"]["alignment_complete"] = True
            else:
                robot_state["fsm_status"]["error_message"] = "Failed to send alignment command"
                transition_to_state("error")
                return
        else:
            print(f"FSM: Already aligned, dx = {dx:.3f}m within tolerance")
            transition_to_state("placing")
            return
    
    # Wait for motor motion to complete
    if is_motor_motion_complete():
        # Re-check alignment after motion
        # Note: In practice, we'd want to re-scan here to get updated dx
        if abs(dx) <= tolerance:
            print(f"FSM: Alignment complete, final dx = {dx:.3f}m")
            transition_to_state("placing")
        else:
            print(f"FSM: Alignment incomplete, retrying. Current dx = {dx:.3f}m")
            robot_state["fsm_status"]["alignment_complete"] = False  # Retry alignment


def fsm_placing_state():
    """
    PLACING STATE: Robot lowers to place shingle at aligned position
    
    Transitions:
        - Motor motion complete → nailing
        - Motor error or timeout → error
    
    Lowers the Z-axis by the placement depth to position the shingle.
    The robot carries a shingle and places it against the roof surface.
    """
    if not robot_state["fsm_status"]["placement_complete"]:
        placement_depth = robot_state["fsm_params"]["placement_depth"]
        print(f"FSM: Starting placement, lowering {placement_depth:.3f}m")
        
        success = nucleo.send_position_command(z_dist=-placement_depth)
        if success:
            robot_state["fsm_status"]["placement_complete"] = True
        else:
            robot_state["fsm_status"]["error_message"] = "Failed to send placement command"
            transition_to_state("error")
            return
    
    # Wait for placement motion to complete
    if is_motor_motion_complete():
        print("FSM: Placement complete, proceeding to nailing")
        transition_to_state("nailing")


def fsm_nailing_state():
    """
    NAILING STATE: Robot activates nail gun to secure shingle
    
    Transitions:
        - Nail duration complete → retracting
        - Nail gun error or timeout → error
    
    Activates the nail gun for the specified duration to secure the shingle.
    This is a timed operation - the nail gun is activated and then deactivated.
    """
    if not robot_state["fsm_status"]["nail_complete"]:
        print("FSM: Activating nail gun")
        success = nucleo.send_position_command(nail=True)
        if success:
            robot_state["fsm_status"]["nail_complete"] = True
        else:
            robot_state["fsm_status"]["error_message"] = "Failed to activate nail gun"
            transition_to_state("error")
            return
    
    # Wait for nail duration to complete
    elapsed = time.time() - robot_state["fsm_status"]["state_start_time"]
    nail_duration = robot_state["fsm_params"]["nail_duration"]
    
    if elapsed >= nail_duration:
        print("FSM: Nailing complete, proceeding to retract")
        # Deactivate nail gun
        nucleo.send_position_command(nail=False)
        transition_to_state("retracting")


def fsm_retracting_state():
    """
    RETRACTING STATE: Robot raises Z-axis back to scanning height
    
    Transitions:
        - Motor motion complete → advancing
        - Motor error or timeout → error
    
    Lifts the Z-axis back to the scanning height after shingle placement.
    This prepares the robot for the next advancement move.
    """
    if not robot_state["fsm_status"].get("retract_complete", False):
        placement_depth = robot_state["fsm_params"]["placement_depth"]
        print(f"FSM: Retracting, raising {placement_depth:.3f}m")
        
        success = nucleo.send_position_command(z_dist=placement_depth)
        if success:
            robot_state["fsm_status"]["retract_complete"] = True
        else:
            robot_state["fsm_status"]["error_message"] = "Failed to send retract command"
            transition_to_state("error")
            return
    
    # Wait for retraction motion to complete
    if is_motor_motion_complete():
        print("FSM: Retraction complete, proceeding to advance")
        transition_to_state("advancing")


def fsm_advancing_state():
    """
    ADVANCING STATE: Robot moves forward to next shingle position
    
    Transitions:
        - Motor motion complete → scanning (for next shingle)
        - Motor error or timeout → error
    
    Moves the Y-axis forward by the advance distance to position for the next shingle.
    This creates the overlapping pattern typical of shingle installation.
    """
    if not robot_state["fsm_status"].get("advance_complete", False):
        advance_distance = robot_state["fsm_params"]["advance_distance"]
        print(f"FSM: Advancing {advance_distance:.3f}m for next shingle")
        
        success = nucleo.send_position_command(y_dist=advance_distance)
        if success:
            robot_state["fsm_status"]["advance_complete"] = True
        else:
            robot_state["fsm_status"]["error_message"] = "Failed to send advance command"
            transition_to_state("error")
            return
    
    # Wait for advance motion to complete
    if is_motor_motion_complete():
        print("FSM: Advance complete, starting next shingle cycle")
        robot_state["shingle_count"] += 1
        transition_to_state("scanning")


def fsm_homing_state():
    """
    HOMING STATE: Robot returns all axes to home position
    
    Transitions:
        - Homing complete → idle
        - Homing error or timeout → error
    
    Moves all axes to their home positions for calibration or shutdown.
    This is typically used for startup calibration or safe shutdown.
    """
    if not robot_state["fsm_status"].get("homing_started", False):
        print("FSM: Starting homing sequence")
        success = nucleo.home_all_axes()
        if success:
            robot_state["fsm_status"]["homing_started"] = True
        else:
            robot_state["fsm_status"]["error_message"] = "Failed to start homing"
            transition_to_state("error")
            return
    
    # Wait for homing to complete (all motors stopped)
    if is_motor_motion_complete():
        print("FSM: Homing complete, returning to idle")
        # Reset all position tracking
        robot_state["motor_status"]["x_position"] = 0.0
        robot_state["motor_status"]["y_position"] = 0.0
        robot_state["motor_status"]["z_position"] = 0.0
        transition_to_state("idle")


def fsm_error_state():
    """
    ERROR STATE: Robot has encountered an error and requires intervention
    
    Transitions:
        - Manual command "clear_error" → idle
        - Manual command "emergency_stop" → emergency_stop
        - Manual command "home" → homing
    
    All autonomous operation stops. The robot waits for manual intervention.
    Error message is available in robot_state["fsm_status"]["error_message"].
    """
    # Error state is reactive - recovery happens via manual commands
    # Stop all motors as a safety measure
    if not robot_state["fsm_status"].get("motors_stopped", False):
        nucleo.emergency_stop()
        robot_state["fsm_status"]["motors_stopped"] = True
        print(f"FSM: Error state - {robot_state['fsm_status']['error_message']}")


def fsm_emergency_stop_state():
    """
    EMERGENCY STOP STATE: All motion immediately halted
    
    Transitions:
        - Manual command "clear_emergency" → idle
        - Manual command "home" → homing
    
    Highest priority state. All motor motion is immediately stopped.
    System remains in this state until manually cleared by operator.
    """
    # Emergency stop is reactive - only manual commands can exit this state
    if not robot_state["fsm_status"].get("emergency_executed", False):
        nucleo.emergency_stop()
        robot_state["fsm_status"]["emergency_executed"] = True
        print("FSM: EMERGENCY STOP ACTIVATED")


def fsm_manual_mode_state():
    """
    MANUAL MODE STATE: Robot accepts direct movement commands
    
    Transitions:
        - Manual command "auto_mode" → idle
        - Manual command "emergency_stop" → emergency_stop
    
    Disables autonomous FSM progression. Robot only responds to direct
    movement commands from the operator interface.
    """
    # Manual mode is reactive - robot only responds to direct commands
    pass


# FSM State Dispatch Table
# Maps state names to their corresponding handler functions
FSM_STATES = {
    "idle": fsm_idle_state,
    "scanning": fsm_scanning_state,
    "aligning": fsm_aligning_state,
    "placing": fsm_placing_state,
    "nailing": fsm_nailing_state,
    "retracting": fsm_retracting_state,
    "advancing": fsm_advancing_state,
    "homing": fsm_homing_state,
    "error": fsm_error_state,
    "emergency_stop": fsm_emergency_stop_state,
    "manual_mode": fsm_manual_mode_state
}


def fsm_main_loop():
    """
    Main FSM execution loop - runs continuously in background thread
    
    This function:
    1. Checks for state timeouts and transitions to error if needed
    2. Dispatches to the appropriate state handler function
    3. Handles any exceptions that occur during state processing
    4. Provides consistent timing and error handling
    
    The FSM runs at 10Hz (100ms cycle time) to provide responsive control
    while not overwhelming the system with excessive processing.
    """
    while True:
        try:
            current_state = robot_state["fsm_state"]
            
            # Skip FSM processing if manual override is enabled
            if robot_state["fsm_params"]["manual_override"] and current_state not in ["emergency_stop", "error"]:
                time.sleep(0.1)
                continue
            
            # Check for state timeout (except in idle, error, and emergency states)
            if current_state not in ["idle", "error", "emergency_stop", "manual_mode"]:
                if check_state_timeout():
                    print(f"FSM: State {current_state} timed out")
                    transition_to_state("error")
                    continue
            
            # Dispatch to appropriate state handler
            if current_state in FSM_STATES:
                FSM_STATES[current_state]()
            else:
                print(f"FSM: Unknown state '{current_state}', transitioning to error")
                robot_state["fsm_status"]["error_message"] = f"Unknown state: {current_state}"
                transition_to_state("error")
            
        except Exception as e:
            print(f"FSM: Exception in state {robot_state['fsm_state']}: {e}")
            robot_state["fsm_status"]["error_message"] = f"Exception: {str(e)}"
            transition_to_state("error")
        
        # FSM cycle timing - 10Hz execution rate
        time.sleep(0.1)


# Start FSM main loop in background thread
fsm_thread = threading.Thread(target=fsm_main_loop, daemon=True)
fsm_thread.start()

def gen_frames():
    while True:
        if SIMULATION_MODE:
            # Generate simulated frame
            try:
                processed_frame, result = process_simulated_frame()
                
                # Update robot state with simulated vision results
                robot_state["dx"] = result.get("dx", 0.0)
                robot_state["detected_lines"] = result.get("detected_lines", 0)
                robot_state["alignment_ok"] = result.get("alignment_ok", False)
                robot_state["confidence"] = result.get("confidence", 0.0)
                robot_state["debug_info"] = result.get("debug_info", {})
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
                time.sleep(0.1)  # Simulate camera frame rate
                
            except Exception as e:
                print(f"Simulation frame generation error: {e}")
                time.sleep(1.0)
        else:
            # Real camera processing
            success, frame = camera.read()
            if not success:
                break
            else:
                processed_frame, result = process_frame(frame)

                # Update robot state with vision results
                robot_state["dx"] = result.get("dx", 0.0)
                robot_state["detected_lines"] = result.get("detected_lines", 0)
                robot_state["alignment_ok"] = result.get("alignment_ok", False)
                robot_state["confidence"] = result.get("confidence", 0.0)
                robot_state["debug_info"] = result.get("debug_info", {})

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/status")
def status():
    return jsonify(robot_state)

@app.route("/command", methods=["POST"])
def command():
    """
    Handle manual commands from web interface
    
    Supports FSM state transitions and direct motor control commands.
    Commands can override automatic FSM progression when needed.
    """
    data = request.get_json()
    cmd = data.get("cmd")
    
    # Get optional parameters for direct movement commands
    x_dist = data.get("x_dist", 0.0)
    y_dist = data.get("y_dist", 0.0)
    z_dist = data.get("z_dist", 0.0)

    # FSM State Transition Commands
    if cmd == "start":
        # Begin autonomous shingle placement cycle
        transition_to_state("scanning")
        
    elif cmd == "stop":
        # Return to idle state
        transition_to_state("idle")
        
    elif cmd == "home":
        # Start homing sequence
        transition_to_state("homing")
        
    elif cmd == "emergency_stop":
        # Immediate emergency stop
        transition_to_state("emergency_stop")
        
    elif cmd == "clear_error":
        # Clear error state and return to idle
        if robot_state["fsm_state"] == "error":
            transition_to_state("idle")
        else:
            return jsonify({"error": "Robot is not in error state"}), 400
            
    elif cmd == "clear_emergency":
        # Clear emergency stop and return to idle
        if robot_state["fsm_state"] == "emergency_stop":
            transition_to_state("idle")
        else:
            return jsonify({"error": "Robot is not in emergency stop"}), 400
    
    # Manual Control Commands
    elif cmd == "manual_mode":
        # Switch to manual control mode
        robot_state["fsm_params"]["manual_override"] = True
        transition_to_state("manual_mode")
        
    elif cmd == "auto_mode":
        # Switch back to automatic mode
        robot_state["fsm_params"]["manual_override"] = False
        transition_to_state("idle")
        
    elif cmd == "move":
        # Direct movement command (works in manual mode or idle)
        if robot_state["fsm_state"] in ["idle", "manual_mode"]:
            success = nucleo.send_position_command(
                x_dist=x_dist, y_dist=y_dist, z_dist=z_dist
            )
            if not success:
                return jsonify({"error": "Failed to send move command"}), 500
        else:
            return jsonify({"error": f"Cannot move in state: {robot_state['fsm_state']}"}), 400
    
    # Legacy Commands (for backward compatibility)
    elif cmd == "place_shingle":
        # Trigger manual placement sequence
        if robot_state["fsm_state"] == "idle":
            transition_to_state("placing")
        else:
            return jsonify({"error": f"Cannot place shingle in state: {robot_state['fsm_state']}"}), 400
            
    elif cmd == "nail":
        # Trigger manual nailing
        if robot_state["fsm_state"] == "idle":
            transition_to_state("nailing")
        else:
            return jsonify({"error": f"Cannot nail in state: {robot_state['fsm_state']}"}), 400
    
    # Parameter Adjustment Commands
    elif cmd == "set_params":
        # Update FSM parameters
        params = data.get("params", {})
        for key, value in params.items():
            if key in robot_state["fsm_params"]:
                robot_state["fsm_params"][key] = value
        
    else:
        return jsonify({"error": f"Unknown command: {cmd}"}), 400

    return jsonify({"success": True, "new_state": robot_state})

# ✅ This must be above the app.run() line!
@app.route("/")
def dashboard():
    return send_from_directory("static", "index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/vision/calibrate", methods=["POST"])
def calibrate_vision():
    """
    Calibrate vision system with 4 corner points
    
    Expected JSON format:
    {
        "corner_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
    Points should be in order: top_left, top_right, bottom_left, bottom_right
    """
    data = request.get_json()
    corner_points = data.get("corner_points")
    
    if not corner_points or len(corner_points) != 4:
        return jsonify({"error": "Exactly 4 corner points required"}), 400
    
    try:
        calibrate_vision_system(corner_points)
        return jsonify({"success": True, "message": "Vision system calibrated"})
    except Exception as e:
        return jsonify({"error": f"Calibration failed: {str(e)}"}), 500


@app.route("/vision/parameters", methods=["GET"])
def get_vision_params():
    """Get current vision system parameters"""
    try:
        params = get_vision_parameters()
        return jsonify({"success": True, "parameters": params})
    except Exception as e:
        return jsonify({"error": f"Failed to get parameters: {str(e)}"}), 500


@app.route("/vision/parameters", methods=["POST"])
def set_vision_params():
    """
    Update vision system parameters
    
    Expected JSON format:
    {
        "parameters": {
            "canny_low": 50,
            "canny_high": 150,
            "alignment_tolerance": 10,
            ...
        }
    }
    """
    data = request.get_json()
    new_params = data.get("parameters", {})
    
    try:
        set_vision_parameters(new_params)
        return jsonify({"success": True, "message": "Parameters updated"})
    except Exception as e:
        return jsonify({"error": f"Failed to update parameters: {str(e)}"}), 500


@app.route("/vision/presets", methods=["GET"])
def get_vision_presets_endpoint():
    """Get all available vision parameter presets"""
    try:
        presets = get_vision_presets()
        return jsonify({"success": True, "presets": presets})
    except Exception as e:
        return jsonify({"error": f"Failed to get presets: {str(e)}"}), 500


@app.route("/vision/presets", methods=["POST"])
def save_vision_preset_endpoint():
    """
    Save current or specified parameters as a preset
    
    Expected JSON format:
    {
        "name": "outdoor_sunny",
        "parameters": {  // optional - if not provided, uses current params
            "canny_low": 50,
            "canny_high": 150,
            ...
        }
    }
    """
    data = request.get_json()
    name = data.get("name")
    parameters = data.get("parameters")
    
    if not name:
        return jsonify({"error": "Preset name is required"}), 400
    
    try:
        success = save_vision_preset(name, parameters)
        if success:
            return jsonify({"success": True, "message": f"Preset '{name}' saved successfully"})
        else:
            return jsonify({"error": f"Failed to save preset '{name}'"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to save preset: {str(e)}"}), 500


@app.route("/vision/presets/<preset_name>", methods=["POST"])
def load_vision_preset_endpoint(preset_name: str):
    """Load a vision parameter preset"""
    try:
        success = load_vision_preset(preset_name)
        if success:
            return jsonify({"success": True, "message": f"Preset '{preset_name}' loaded successfully"})
        else:
            return jsonify({"error": f"Failed to load preset '{preset_name}'"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load preset: {str(e)}"}), 500


@app.route("/vision/presets/<preset_name>", methods=["DELETE"])
def delete_vision_preset_endpoint(preset_name: str):
    """Delete a vision parameter preset"""
    try:
        success = delete_vision_preset(preset_name)
        if success:
            return jsonify({"success": True, "message": f"Preset '{preset_name}' deleted successfully"})
        else:
            return jsonify({"error": f"Failed to delete preset '{preset_name}'"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to delete preset: {str(e)}"}), 500


@app.route("/simulation/status", methods=["GET"])
def simulation_status():
    """Get current simulation mode status"""
    return jsonify({
        "simulation_mode": SIMULATION_MODE,
        "config": get_simulation_config().__dict__ if is_simulation_mode() else None
    })


@app.route("/simulation/config", methods=["GET"])
def get_simulation_config_endpoint():
    """Get current simulation configuration"""
    if not SIMULATION_MODE:
        return jsonify({"error": "Not in simulation mode"}), 400
    
    try:
        config = get_simulation_config()
        return jsonify({"success": True, "config": config.__dict__})
    except Exception as e:
        return jsonify({"error": f"Failed to get config: {str(e)}"}), 500


@app.route("/simulation/config", methods=["POST"])
def update_simulation_config_endpoint():
    """
    Update simulation configuration
    
    Expected JSON format:
    {
        "config": {
            "motor_speed_multiplier": 5.0,
            "simulated_dx_offset": 0.02,
            "enable_failure_injection": true,
            ...
        }
    }
    """
    if not SIMULATION_MODE:
        return jsonify({"error": "Not in simulation mode"}), 400
    
    data = request.get_json()
    new_config = data.get("config", {})
    
    try:
        update_simulation_config(new_config)
        return jsonify({"success": True, "message": "Simulation config updated"})
    except Exception as e:
        return jsonify({"error": f"Failed to update config: {str(e)}"}), 500


@app.route("/simulation/inject_failure", methods=["POST"])
def inject_failure():
    """
    Inject specific failures for testing
    
    Expected JSON format:
    {
        "failure_type": "motor_error" | "vision_failure" | "connection_loss",
        "duration": 5.0  // seconds (optional)
    }
    """
    if not SIMULATION_MODE:
        return jsonify({"error": "Not in simulation mode"}), 400
    
    data = request.get_json()
    failure_type = data.get("failure_type")
    duration = data.get("duration", 5.0)
    
    try:
        # This would be implemented to inject specific failures
        # For now, just enable failure injection temporarily
        current_config = get_simulation_config()
        original_failure_rate = current_config.motor_failure_rate
        
        if failure_type == "motor_error":
            update_simulation_config({"motor_failure_rate": 1.0})
            # Reset after duration (would need a timer in real implementation)
        elif failure_type == "vision_failure":
            update_simulation_config({"detection_failure_rate": 1.0})
        elif failure_type == "connection_loss":
            update_simulation_config({"connection_stable": False})
        
        return jsonify({"success": True, "message": f"Injected {failure_type} for {duration}s"})
    except Exception as e:
        return jsonify({"error": f"Failed to inject failure: {str(e)}"}), 500


@app.route("/simulation/regenerate_scene", methods=["POST"])
def regenerate_scene():
    """Regenerate the synthetic shingle scene"""
    if not SIMULATION_MODE:
        return jsonify({"error": "Not in simulation mode"}), 400
    
    try:
        vision_sim = get_simulated_vision()
        if vision_sim:
            vision_sim._generate_synthetic_scene()
            return jsonify({"success": True, "message": "Scene regenerated"})
        else:
            return jsonify({"error": "Simulation vision not available"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to regenerate scene: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

