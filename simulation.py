"""
Simulation Module for Canopy Robot
Provides mock hardware implementations for testing without physical hardware.
"""

import time
import threading
import numpy as np
import cv2
import json
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation mode"""
    # Motor simulation
    motor_speed_multiplier: float = 5.0  # Speed up movements for testing
    motor_noise_level: float = 0.001  # Position noise in meters
    motor_failure_rate: float = 0.0  # Probability of motor failures
    
    # Vision simulation
    simulated_dx_offset: float = 0.0  # Inject specific dx for testing
    vision_noise_level: float = 0.005  # Vision measurement noise
    detection_failure_rate: float = 0.0  # Probability of vision failures
    synthetic_lines_count: int = 3  # Number of synthetic shingle lines
    
    # System simulation
    connection_stable: bool = True  # Simulate connection issues
    enable_failure_injection: bool = False  # Enable random failures
    
    def save(self, filename: str = "simulation_config.json"):
        """Save configuration to file"""
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filename: str = "simulation_config.json"):
        """Load configuration from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    return cls(**data)
        except Exception as e:
            print(f"Failed to load simulation config: {e}")
        return cls()


class SimulatedNucleoComms:
    """
    Simulated STM32 Nucleo communication for testing
    Mimics the interface of NucleoComms but with simulated behavior
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.connected = True
        self.lock = threading.Lock()
        
        # Simulated motor state
        self.motor_positions = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.motor_targets = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.motor_running = {"x": False, "y": False, "z": False}
        self.motor_speeds = {"x": 0.05, "y": 0.05, "z": 0.02}  # m/s
        
        # Nail gun and vacuum state
        self.nail_active = False
        self.vacuum_active = False
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        print("🤖 Simulated Nucleo initialized")
    
    def connect(self) -> bool:
        """Simulate connection establishment"""
        if not self.config.connection_stable:
            return np.random.random() > 0.1  # 90% success rate
        
        time.sleep(0.5)  # Simulate connection delay
        self.connected = True
        print("🤖 Simulated Nucleo connected")
        return True
    
    def disconnect(self):
        """Simulate disconnection"""
        self.connected = False
        print("🤖 Simulated Nucleo disconnected")
    
    def is_connected(self) -> bool:
        """Check simulated connection status"""
        if not self.config.connection_stable:
            # Occasionally drop connection for testing
            if np.random.random() < 0.001:  # 0.1% chance per call
                self.connected = False
        return self.connected
    
    def send_position_command(self, x_dist: float = 0.0, y_dist: float = 0.0, 
                            z_dist: float = 0.0, r_dist: float = 0.0, 
                            n_dist: float = 0.0, nail: bool = False, 
                            vacuum: bool = False, calibrate: bool = False) -> bool:
        """Simulate position command"""
        if not self.is_connected():
            return False
        
        # Simulate command failure
        if self.config.enable_failure_injection and np.random.random() < self.config.motor_failure_rate:
            print("🤖 Simulated motor command failure")
            return False
        
        with self.lock:
            # Update targets for position movements
            if x_dist != 0.0:
                self.motor_targets["x"] = self.motor_positions["x"] + x_dist
                self.motor_running["x"] = True
            
            if y_dist != 0.0:
                self.motor_targets["y"] = self.motor_positions["y"] + y_dist
                self.motor_running["y"] = True
            
            if z_dist != 0.0:
                self.motor_targets["z"] = self.motor_positions["z"] + z_dist
                self.motor_running["z"] = True
            
            # Handle nail gun and vacuum
            self.nail_active = nail
            self.vacuum_active = vacuum
            
            # Handle homing
            if calibrate:
                print("🤖 Simulated homing started")
                self.motor_targets = {"x": 0.0, "y": 0.0, "z": 0.0}
                self.motor_running = {"x": True, "y": True, "z": True}
        
        return True
    
    def home_all_axes(self) -> bool:
        """Simulate homing all axes"""
        return self.send_position_command(calibrate=True)
    
    def emergency_stop(self) -> bool:
        """Simulate emergency stop"""
        if not self.is_connected():
            return False
        
        with self.lock:
            # Stop all motion immediately
            self.motor_running = {"x": False, "y": False, "z": False}
            self.motor_targets = self.motor_positions.copy()
            self.nail_active = False
            self.vacuum_active = False
        
        print("🤖 Simulated emergency stop")
        return True
    
    def read_motor_status(self) -> Optional[Dict]:
        """Return simulated motor status"""
        if not self.is_connected():
            return None
        
        with self.lock:
            # Add some noise to positions
            noise_level = self.config.motor_noise_level
            positions = {}
            for axis in ["x", "y", "z"]:
                noise = np.random.normal(0, noise_level)
                positions[f"{axis}_position"] = self.motor_positions[axis] + noise
                positions[f"{axis}_running"] = self.motor_running[axis]
        
        return positions
    
    def _simulation_loop(self):
        """Background thread to simulate motor movements"""
        while True:
            try:
                if not self.connected:
                    time.sleep(0.1)
                    continue
                
                with self.lock:
                    dt = 0.1 * self.config.motor_speed_multiplier  # Time step
                    
                    for axis in ["x", "y", "z"]:
                        if self.motor_running[axis]:
                            current = self.motor_positions[axis]
                            target = self.motor_targets[axis]
                            speed = self.motor_speeds[axis]
                            
                            # Calculate movement
                            distance = target - current
                            max_move = speed * dt
                            
                            if abs(distance) <= max_move:
                                # Reached target
                                self.motor_positions[axis] = target
                                self.motor_running[axis] = False
                                print(f"🤖 Motor {axis.upper()} reached target: {target:.3f}m")
                            else:
                                # Move towards target
                                direction = 1 if distance > 0 else -1
                                self.motor_positions[axis] += direction * max_move
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"🤖 Simulation loop error: {e}")
                time.sleep(1.0)


class SimulatedVisionSystem:
    """
    Simulated vision system for testing
    Generates synthetic camera frames with controllable shingle patterns
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.frame_width = 640
        self.frame_height = 480
        self.synthetic_edges = []
        self._generate_synthetic_scene()
        
        print("👁️ Simulated vision system initialized")
    
    def _generate_synthetic_scene(self):
        """Generate synthetic shingle edges for testing"""
        self.synthetic_edges = []
        
        # Create horizontal lines representing shingle edges
        base_y = self.frame_height // 2
        line_spacing = 60
        
        for i in range(self.config.synthetic_lines_count):
            y = base_y + (i - 1) * line_spacing
            # Rightmost edge with some variation
            x_end = 400 + np.random.normal(0, 20)
            x_start = x_end - 150 - np.random.normal(0, 10)
            
            self.synthetic_edges.append({
                "x1": max(0, int(x_start)),
                "y1": int(y),
                "x2": min(self.frame_width, int(x_end)),
                "y2": int(y + np.random.normal(0, 3))  # Slight angle variation
            })
    
    def generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic camera frame with simulated shingles"""
        # Create base frame (roof background)
        frame = np.random.randint(80, 120, (self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add some texture
        noise = np.random.randint(-30, 30, (self.frame_height, self.frame_width, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Draw synthetic shingle edges
        for edge in self.synthetic_edges:
            # Add thickness variation
            thickness = np.random.randint(2, 6)
            color = (np.random.randint(40, 80), np.random.randint(40, 80), np.random.randint(40, 80))
            
            cv2.line(frame, (edge["x1"], edge["y1"]), (edge["x2"], edge["y2"]), color, thickness)
        
        # Apply some blur to make it more realistic
        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
        
        return frame
    
    def calculate_simulated_dx(self) -> Tuple[float, Dict]:
        """Calculate simulated dx value with controllable offset"""
        # Base dx from rightmost synthetic edge
        if self.synthetic_edges:
            rightmost_edge = max(self.synthetic_edges, key=lambda e: max(e["x1"], e["x2"]))
            rightmost_x = max(rightmost_edge["x1"], rightmost_edge["x2"])
            
            # Expected position (configurable)
            expected_x = 400
            dx_pixels = rightmost_x - expected_x
            
            # Add configured offset for testing
            dx_pixels += self.config.simulated_dx_offset * 100  # Convert m to pixels
            
            # Add noise
            noise = np.random.normal(0, self.config.vision_noise_level * 100)
            dx_pixels += noise
            
            # Convert to meters (assuming 1000 pixels per meter)
            dx_meters = dx_pixels / 1000.0
            
            # Simulate detection failure
            if self.config.enable_failure_injection and np.random.random() < self.config.detection_failure_rate:
                return 0.0, {"error": "Simulated vision failure", "confidence": 0.0}
            
            debug_info = {
                "detected_lines": len(self.synthetic_edges),
                "expected_x_pixel": expected_x,
                "detected_x_pixel": rightmost_x,
                "dx_pixels": dx_pixels,
                "confidence": 0.8 + np.random.normal(0, 0.1)
            }
            
            return dx_meters, debug_info
        
        return 0.0, {"error": "No synthetic edges", "confidence": 0.0}


# Global simulation instances
_simulation_config = None
_simulated_nucleo = None
_simulated_vision = None


def initialize_simulation(config_file: str = "simulation_config.json"):
    """Initialize simulation mode with configuration"""
    global _simulation_config, _simulated_nucleo, _simulated_vision
    
    try:
        _simulation_config = SimulationConfig.load(config_file)
        _simulated_nucleo = SimulatedNucleoComms(_simulation_config)
        _simulated_vision = SimulatedVisionSystem(_simulation_config)
        
        print("🚀 Simulation mode initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_simulated_nucleo():
    """Get the simulated Nucleo communications instance"""
    return _simulated_nucleo


def get_simulated_vision():
    """Get the simulated vision system instance"""
    return _simulated_vision


def get_simulation_config():
    """Get the current simulation configuration"""
    return _simulation_config


def update_simulation_config(new_config: Dict):
    """Update simulation configuration parameters"""
    global _simulation_config
    if _simulation_config:
        for key, value in new_config.items():
            if hasattr(_simulation_config, key):
                setattr(_simulation_config, key, value)
        _simulation_config.save()
        print(f"🔧 Simulation config updated: {new_config}")


def is_simulation_mode() -> bool:
    """Check if currently running in simulation mode"""
    return _simulation_config is not None


# Simulation-specific vision processing function
def process_simulated_frame() -> Tuple[np.ndarray, Dict]:
    """Process simulated frame and return synthetic results"""
    if not _simulated_vision:
        raise RuntimeError("Simulation not initialized")
    
    # Generate synthetic frame
    frame = _simulated_vision.generate_synthetic_frame()
    
    # Calculate simulated dx
    dx_meters, debug_info = _simulated_vision.calculate_simulated_dx()
    
    # Create visualization (copy original vision processing overlay logic)
    vis_frame = frame.copy()
    
    # Draw synthetic detection overlays
    for edge in _simulated_vision.synthetic_edges:
        cv2.line(vis_frame, (edge["x1"], edge["y1"]), (edge["x2"], edge["y2"]), (0, 255, 0), 3)
    
    # Draw expected and detected positions
    expected_x = debug_info.get("expected_x_pixel", 400)
    detected_x = debug_info.get("detected_x_pixel", expected_x)
    
    cv2.line(vis_frame, (expected_x, 0), (expected_x, vis_frame.shape[0]), (255, 0, 0), 2)
    cv2.line(vis_frame, (int(detected_x), 0), (int(detected_x), vis_frame.shape[0]), (0, 0, 255), 2)
    
    # Add simulation indicator
    cv2.putText(vis_frame, "SIMULATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(vis_frame, f"dx: {dx_meters*1000:.1f}mm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    result = {
        "dx": dx_meters,
        "alignment_ok": abs(dx_meters) < 0.01,  # 1cm tolerance
        "detected_lines": len(_simulated_vision.synthetic_edges),
        "confidence": debug_info.get("confidence", 0.8),
        "debug_info": debug_info,
        "simulation": True
    }
    
    return vis_frame, result