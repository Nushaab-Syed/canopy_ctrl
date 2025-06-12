"""
STM32 Nucleo Communication Module
Handles serial communication between Raspberry Pi and STM32 Nucleo for motor control.
Based on existing communication protocol with structured data format.
"""

import serial
import struct
import time
import threading
from typing import Dict, Optional, Tuple


class NucleoComms:
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200):
        """
        Initialize communication with STM32 Nucleo
        
        Args:
            port: Serial port (default '/dev/ttyACM0' for Raspberry Pi)
            baudrate: Communication speed (default 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        
        # Motor parameters
        self.lead_screw_pitch = 0.008  # meters per revolution
        self.steps_per_rev = 6400      # microsteps per revolution
        
        # Struct formats from original code
        self.setpoint_struct_format = 'iiiiiBBBx'  # z,y,x,r,n steps + nail,vacuum,cal flags
        self.motor_data_struct_format = 'iBxxxiBxxxiBxxxiBxxxiBxxx'  # motor feedback format
        
        self.setpoint_struct_size = struct.calcsize(self.setpoint_struct_format)
        self.motor_data_struct_size = struct.calcsize(self.motor_data_struct_format)
        
        # Thread safety
        self.lock = threading.Lock()
        
    def connect(self) -> bool:
        """
        Establish serial connection to STM32 Nucleo
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow connection to stabilize
            self.connected = True
            print(f"Connected to Nucleo on {self.ser.name}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to Nucleo: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print("Disconnected from Nucleo")
    
    def distance_to_steps(self, distance_m: float) -> int:
        """
        Convert distance in meters to motor steps
        
        Args:
            distance_m: Distance in meters
            
        Returns:
            int: Number of motor steps
        """
        return int(distance_m / self.lead_screw_pitch * self.steps_per_rev)
    
    def steps_to_distance(self, steps: int) -> float:
        """
        Convert motor steps to distance in meters
        
        Args:
            steps: Number of motor steps
            
        Returns:
            float: Distance in meters
        """
        return float(steps) / self.steps_per_rev * self.lead_screw_pitch
    
    def send_position_command(self, 
                            x_dist: float = 0.0, 
                            y_dist: float = 0.0, 
                            z_dist: float = 0.0,
                            r_dist: float = 0.0, 
                            n_dist: float = 0.0,
                            nail: bool = False, 
                            vacuum: bool = False, 
                            calibrate: bool = False) -> bool:
        """
        Send position command to STM32 Nucleo
        
        Args:
            x_dist, y_dist, z_dist, r_dist, n_dist: Distances in meters
            nail: Activate nail gun
            vacuum: Activate vacuum
            calibrate: Trigger calibration
            
        Returns:
            bool: True if command sent successfully
        """
        if not self.connected or not self.ser:
            print("Not connected to Nucleo")
            return False
        
        try:
            with self.lock:
                # Convert distances to steps
                x_steps = self.distance_to_steps(x_dist)
                y_steps = self.distance_to_steps(y_dist)
                z_steps = self.distance_to_steps(z_dist)
                r_steps = self.distance_to_steps(r_dist)
                n_steps = self.distance_to_steps(n_dist)
                
                # Pack data according to struct format
                position_data = (
                    z_steps, y_steps, x_steps, r_steps, n_steps,
                    int(nail), int(vacuum), int(calibrate)
                )
                
                # Send header byte and packed data
                self.ser.write(b')')
                self.ser.write(struct.pack(self.setpoint_struct_format, *position_data))
                
                print(f"Sent position command: x={x_dist:.3f}m, y={y_dist:.3f}m, z={z_dist:.3f}m")
                return True
                
        except Exception as e:
            print(f"Error sending position command: {e}")
            return False
    
    def read_motor_status(self) -> Optional[Dict]:
        """
        Read motor status feedback from STM32 Nucleo
        
        Returns:
            Dict with motor positions and run flags, or None if error
        """
        if not self.connected or not self.ser:
            return None
        
        try:
            with self.lock:
                # Check if data is available
                if self.ser.in_waiting < self.motor_data_struct_size:
                    return None
                
                # Read motor data
                data = self.ser.read(self.motor_data_struct_size)
                if len(data) != self.motor_data_struct_size:
                    return None
                
                # Unpack according to struct format
                unpacked_data = struct.unpack(self.motor_data_struct_format, data)
                
                # Convert to readable format
                motor_status = {
                    'z_position': self.steps_to_distance(unpacked_data[0]),
                    'z_running': bool(unpacked_data[1]),
                    'y_position': self.steps_to_distance(unpacked_data[2]),
                    'y_running': bool(unpacked_data[3]),
                    'x_position': self.steps_to_distance(unpacked_data[4]),
                    'x_running': bool(unpacked_data[5]),
                    'r_position': self.steps_to_distance(unpacked_data[6]),
                    'r_running': bool(unpacked_data[7]),
                    'n_position': self.steps_to_distance(unpacked_data[8]),
                    'n_running': bool(unpacked_data[9])
                }
                
                return motor_status
                
        except Exception as e:
            print(f"Error reading motor status: {e}")
            return None
    
    def home_all_axes(self) -> bool:
        """
        Send command to home all motor axes
        
        Returns:
            bool: True if command sent successfully
        """
        return self.send_position_command(calibrate=True)
    
    def emergency_stop(self) -> bool:
        """
        Send emergency stop command
        
        Returns:
            bool: True if command sent successfully
        """
        return self.send_position_command(0, 0, 0, 0, 0, False, False, False)
    
    def is_connected(self) -> bool:
        """Check if connected to Nucleo"""
        return self.connected and self.ser and self.ser.is_open


# Convenience functions for common operations
def create_nucleo_connection(port: str = '/dev/ttyACM0') -> Optional[NucleoComms]:
    """
    Create and test Nucleo connection
    
    Args:
        port: Serial port path
        
    Returns:
        NucleoComms instance if successful, None otherwise
    """
    nucleo = NucleoComms(port)
    if nucleo.connect():
        return nucleo
    return None


if __name__ == "__main__":
    # Test the communication module
    print("Testing Nucleo communication...")
    
    nucleo = create_nucleo_connection()
    if nucleo:
        print("Connection established!")
        
        # Test position command
        success = nucleo.send_position_command(x_dist=-0.25, y_dist=-0.3, z_dist=-0.01)
        if success:
            print("Position command sent successfully")
            
            # Try to read motor status
            time.sleep(0.5)
            status = nucleo.read_motor_status()
            if status:
                print(f"Motor status: {status}")
            else:
                print("No motor status received")
        
        nucleo.disconnect()
    else:
        print("Failed to establish connection")