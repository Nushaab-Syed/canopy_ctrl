import cv2
import numpy as np
import json
import os
from typing import Tuple, Dict, Optional, List


class ShingleVisionSystem:
    """
    Computer vision system for roofing robot shingle alignment detection
    
    Uses perspective transform and edge detection to calculate lateral offset (dx)
    needed for precise shingle placement alignment.
    """
    
    def __init__(self, calibration_file: str = "vision_calibration.json"):
        self.calibration_file = calibration_file
        self.presets_file = "vision_presets.json"
        self.homography_matrix = None
        self.roi_bounds = None  # Region of interest for shingle detection
        
        # Vision processing parameters
        self.params = {
            # Perspective transform
            "image_width": 640,
            "image_height": 480,
            "rectified_width": 400,
            "rectified_height": 300,
            
            # Edge detection
            "gaussian_kernel": (5, 5),
            "gaussian_sigma": 1.0,
            "canny_low": 50,
            "canny_high": 150,
            "morph_kernel_size": 3,
            
            # Hough line detection
            "hough_rho": 1,
            "hough_theta": np.pi/180,
            "hough_threshold": 50,
            "min_line_length": 50,
            "max_line_gap": 10,
            
            # Shingle detection
            "expected_shingle_x": 200,  # Expected x position of new shingle edge (pixels)
            "alignment_tolerance": 10,   # Pixel tolerance for alignment
            "min_edge_length": 80,      # Minimum length for valid shingle edge
            "max_slope_deg": 15,        # Max angle deviation from horizontal (degrees)
            
            # Calibration points (pixels in original image)
            "calibration_points": {
                "top_left": [100, 100],
                "top_right": [540, 100], 
                "bottom_left": [80, 380],
                "bottom_right": [560, 380]
            },
            
            # Real-world dimensions (meters)
            "real_world_width": 0.4,    # 40cm width
            "real_world_height": 0.3,   # 30cm height
            "pixels_per_meter": 1000    # Calculated from calibration
        }
        
        self.load_calibration()
        self.load_presets()
    
    def load_calibration(self) -> bool:
        """Load perspective transform calibration from file"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    cal_data = json.load(f)
                    
                # Convert homography matrix back from list
                if "homography_matrix" in cal_data:
                    self.homography_matrix = np.array(cal_data["homography_matrix"])
                    print("Loaded calibration from file")
                    return True
                    
        except Exception as e:
            print(f"Failed to load calibration: {e}")
        
        # Generate default calibration if none exists
        self.generate_default_calibration()
        return False
    
    def save_calibration(self):
        """Save current calibration to file"""
        try:
            cal_data = {
                "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
                "params": self.params
            }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(cal_data, f, indent=2)
                
            print(f"Calibration saved to {self.calibration_file}")
            
        except Exception as e:
            print(f"Failed to save calibration: {e}")
    
    def generate_default_calibration(self):
        """Generate a default perspective transform for testing"""
        # Source points from camera view (trapezoid due to perspective)
        src_points = np.float32([
            self.params["calibration_points"]["top_left"],
            self.params["calibration_points"]["top_right"],
            self.params["calibration_points"]["bottom_left"],
            self.params["calibration_points"]["bottom_right"]
        ])
        
        # Destination points (rectified top-down view)
        dst_points = np.float32([
            [0, 0],
            [self.params["rectified_width"], 0],
            [0, self.params["rectified_height"]],
            [self.params["rectified_width"], self.params["rectified_height"]]
        ])
        
        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Calculate pixels per meter for real-world measurements
        pixels_x = self.params["rectified_width"] / self.params["real_world_width"]
        pixels_y = self.params["rectified_height"] / self.params["real_world_height"] 
        self.params["pixels_per_meter"] = (pixels_x + pixels_y) / 2
        
        print("Generated default calibration")
        self.save_calibration()
    
    def load_presets(self):
        """Load vision parameter presets from file"""
        self.presets = {}
        try:
            if os.path.exists(self.presets_file):
                with open(self.presets_file, 'r') as f:
                    self.presets = json.load(f)
                    print(f"Loaded {len(self.presets)} vision presets")
        except Exception as e:
            print(f"Failed to load presets: {e}")
            self.presets = {}
    
    def save_presets(self):
        """Save vision parameter presets to file"""
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
            print(f"Saved {len(self.presets)} vision presets to {self.presets_file}")
        except Exception as e:
            print(f"Failed to save presets: {e}")
    
    def get_presets(self) -> Dict:
        """Get all available presets with metadata"""
        preset_info = {}
        for name, preset_data in self.presets.items():
            preset_info[name] = {
                "params": preset_data.get("params", {}),
                "description": preset_data.get("description", ""),
                "created_date": preset_data.get("created_date", ""),
                "last_modified": preset_data.get("last_modified", "")
            }
        return preset_info
    
    def save_preset(self, name: str, preset_params: Optional[Dict] = None) -> bool:
        """
        Save current or specified parameters as a preset
        
        Args:
            name: Name for the preset
            preset_params: Optional specific parameters to save. If None, uses current params
            
        Returns:
            bool: True if saved successfully
        """
        try:
            import datetime
            now = datetime.datetime.now().isoformat()
            
            params_to_save = preset_params if preset_params is not None else self.params.copy()
            
            # Remove non-serializable items
            serializable_params = {}
            for key, value in params_to_save.items():
                if key == "calibration_points":
                    serializable_params[key] = value
                elif isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_params[key] = value
                elif hasattr(value, 'tolist'):  # numpy arrays
                    serializable_params[key] = value.tolist()
                else:
                    print(f"Skipping non-serializable parameter: {key}")
            
            self.presets[name] = {
                "params": serializable_params,
                "description": f"Vision preset saved on {now[:10]}",
                "created_date": self.presets.get(name, {}).get("created_date", now),
                "last_modified": now
            }
            
            self.save_presets()
            print(f"Saved vision preset: {name}")
            return True
            
        except Exception as e:
            print(f"Failed to save preset {name}: {e}")
            return False
    
    def load_preset(self, name: str) -> bool:
        """
        Load a vision parameter preset
        
        Args:
            name: Name of the preset to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if name not in self.presets:
                print(f"Preset '{name}' not found")
                return False
            
            preset_data = self.presets[name]
            saved_params = preset_data.get("params", {})
            
            # Update current parameters with preset values
            for key, value in saved_params.items():
                if key in self.params:
                    # Convert back to numpy arrays if needed
                    if key == "gaussian_kernel" and isinstance(value, list):
                        self.params[key] = tuple(value)
                    else:
                        self.params[key] = value
            
            # Regenerate calibration if calibration points changed
            if "calibration_points" in saved_params:
                self.generate_default_calibration()
            else:
                self.save_calibration()
            
            print(f"Loaded vision preset: {name}")
            return True
            
        except Exception as e:
            print(f"Failed to load preset {name}: {e}")
            return False
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a vision parameter preset
        
        Args:
            name: Name of the preset to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            if name not in self.presets:
                print(f"Preset '{name}' not found")
                return False
            
            del self.presets[name]
            self.save_presets()
            print(f"Deleted vision preset: {name}")
            return True
            
        except Exception as e:
            print(f"Failed to delete preset {name}: {e}")
            return False
    
    def calibrate_from_points(self, corner_points: List[Tuple[int, int]]):
        """
        Calibrate perspective transform from 4 corner points
        
        Args:
            corner_points: List of 4 (x,y) points in order: top_left, top_right, bottom_left, bottom_right
        """
        if len(corner_points) != 4:
            raise ValueError("Exactly 4 corner points required for calibration")
        
        # Update calibration points
        point_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        for i, point_name in enumerate(point_names):
            self.params["calibration_points"][point_name] = list(corner_points[i])
        
        # Regenerate calibration
        self.generate_default_calibration()
    
    def apply_perspective_transform(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to get top-down rectified view
        
        Args:
            frame: Input BGR frame from camera
            
        Returns:
            Rectified top-down view
        """
        if self.homography_matrix is None:
            return frame
        
        rectified = cv2.warpPerspective(
            frame, 
            self.homography_matrix,
            (self.params["rectified_width"], self.params["rectified_height"])
        )
        
        return rectified
    
    def detect_shingle_edges(self, rectified_frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detect horizontal shingle edges in rectified frame
        
        Args:
            rectified_frame: Top-down rectified BGR frame
            
        Returns:
            Tuple of (detected_lines, edge_image)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(
            gray, 
            self.params["gaussian_kernel"], 
            self.params["gaussian_sigma"]
        )
        
        # Canny edge detection
        edges = cv2.Canny(
            blur,
            self.params["canny_low"],
            self.params["canny_high"]
        )
        
        # Morphological closing to connect edge segments
        kernel = np.ones((self.params["morph_kernel_size"], self.params["morph_kernel_size"]), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            self.params["hough_rho"],
            self.params["hough_theta"],
            self.params["hough_threshold"],
            minLineLength=self.params["min_line_length"],
            maxLineGap=self.params["max_line_gap"]
        )
        
        return lines if lines is not None else [], edges
    
    def filter_horizontal_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter lines to keep only approximately horizontal ones (shingle edges)
        
        Args:
            lines: List of detected lines from Hough transform
            
        Returns:
            Filtered list of horizontal lines
        """
        horizontal_lines = []
        max_slope = np.tan(np.radians(self.params["max_slope_deg"]))
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length and slope
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length < self.params["min_edge_length"]:
                continue
                
            # Calculate slope (avoid division by zero)
            if abs(x2 - x1) < 1:
                slope = float('inf')
            else:
                slope = abs((y2 - y1) / (x2 - x1))
            
            # Keep approximately horizontal lines
            if slope <= max_slope:
                horizontal_lines.append(line)
        
        return horizontal_lines
    
    def calculate_dx_offset(self, horizontal_lines: List[np.ndarray]) -> Tuple[float, bool, Dict]:
        """
        Calculate lateral offset (dx) needed for shingle alignment
        
        Args:
            horizontal_lines: List of detected horizontal lines
            
        Returns:
            Tuple of (dx_meters, alignment_ok, debug_info)
        """
        debug_info = {
            "detected_lines": len(horizontal_lines),
            "expected_x_pixel": self.params["expected_shingle_x"],
            "detected_x_pixel": None,
            "dx_pixels": 0,
            "confidence": 0.0
        }
        
        if not horizontal_lines:
            return 0.0, False, debug_info
        
        # Find the rightmost edge (likely the previous shingle edge)
        rightmost_x = 0
        best_line = None
        
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            
            # Take the rightmost point of the line
            line_right_x = max(x1, x2)
            
            if line_right_x > rightmost_x:
                rightmost_x = line_right_x
                best_line = line
        
        if best_line is not None:
            debug_info["detected_x_pixel"] = rightmost_x
            
            # Calculate pixel offset from expected position
            dx_pixels = rightmost_x - self.params["expected_shingle_x"]
            debug_info["dx_pixels"] = dx_pixels
            
            # Convert to real-world meters
            dx_meters = dx_pixels / self.params["pixels_per_meter"]
            
            # Check if within alignment tolerance
            pixel_tolerance = self.params["alignment_tolerance"]
            alignment_ok = abs(dx_pixels) <= pixel_tolerance
            
            # Calculate confidence based on line quality
            x1, y1, x2, y2 = best_line[0]
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            debug_info["confidence"] = min(1.0, line_length / 100.0)  # Normalize to 0-1
            
            return dx_meters, alignment_ok, debug_info
        
        return 0.0, False, debug_info
    
    def draw_debug_overlay(self, frame: np.ndarray, rectified_frame: np.ndarray, 
                          lines: List[np.ndarray], dx_info: Dict) -> np.ndarray:
        """
        Draw debug information on the frame for visualization
        
        Args:
            frame: Original camera frame
            rectified_frame: Rectified top-down frame
            lines: Detected lines
            dx_info: Debug information from dx calculation
            
        Returns:
            Frame with debug overlays
        """
        # Start with original frame
        overlay_frame = frame.copy()
        
        # Draw calibration points on original frame
        cal_points = self.params["calibration_points"]
        for point_name, (x, y) in cal_points.items():
            cv2.circle(overlay_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(overlay_frame, point_name[:2], (int(x) + 10, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Create side-by-side visualization
        h, w = frame.shape[:2]
        rh, rw = rectified_frame.shape[:2]
        
        # Resize rectified frame to fit alongside original
        scale = h / rh
        new_rw = int(rw * scale)
        rectified_resized = cv2.resize(rectified_frame, (new_rw, h))
        
        # Draw detected lines on rectified view
        rectified_with_lines = rectified_resized.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Scale coordinates
            x1, x2 = int(x1 * scale), int(x2 * scale)
            y1, y2 = int(y1 * scale), int(y2 * scale)
            cv2.line(rectified_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw expected shingle position
        expected_x = int(self.params["expected_shingle_x"] * scale)
        cv2.line(rectified_with_lines, (expected_x, 0), (expected_x, h), (255, 0, 0), 2)
        
        # Draw detected position if available
        if dx_info["detected_x_pixel"] is not None:
            detected_x = int(dx_info["detected_x_pixel"] * scale)
            cv2.line(rectified_with_lines, (detected_x, 0), (detected_x, h), (0, 0, 255), 2)
        
        # Combine frames
        combined = np.hstack([overlay_frame, rectified_with_lines])
        
        # Add text overlay with status
        status_text = [
            f"dx: {dx_info.get('dx_pixels', 0):.1f}px",
            f"Lines: {dx_info.get('detected_lines', 0)}",
            f"Conf: {dx_info.get('confidence', 0):.2f}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(combined, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined


# Global vision system instance
vision_system = ShingleVisionSystem()


def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Main vision processing function called by Flask app
    
    Args:
        frame: Raw BGR frame from camera
        
    Returns:
        Tuple of (processed_frame_with_overlays, result_dict)
    """
    try:
        # Apply perspective transform to get top-down view
        rectified_frame = vision_system.apply_perspective_transform(frame)
        
        # Detect shingle edges in rectified frame
        detected_lines, edge_image = vision_system.detect_shingle_edges(rectified_frame)
        
        # Filter for horizontal lines (shingle edges)
        horizontal_lines = vision_system.filter_horizontal_lines(detected_lines)
        
        # Calculate dx offset for alignment
        dx_meters, alignment_ok, debug_info = vision_system.calculate_dx_offset(horizontal_lines)
        
        # Create visualization with debug overlays
        vis_frame = vision_system.draw_debug_overlay(frame, rectified_frame, horizontal_lines, debug_info)
        
        # Prepare result dictionary
        result = {
            "dx": dx_meters,
            "alignment_ok": alignment_ok,
            "detected_lines": len(horizontal_lines),
            "confidence": debug_info.get("confidence", 0.0),
            "debug_info": debug_info
        }
        
        return vis_frame, result
        
    except Exception as e:
        print(f"Vision processing error: {e}")
        
        # Return error state
        error_frame = frame.copy()
        cv2.putText(error_frame, f"Vision Error: {str(e)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        result = {
            "dx": 0.0,
            "alignment_ok": False,
            "detected_lines": 0,
            "confidence": 0.0,
            "error": str(e)
        }
        
        return error_frame, result


def calibrate_vision_system(corner_points: List[Tuple[int, int]]):
    """
    Calibrate the vision system with 4 corner points
    
    Args:
        corner_points: List of 4 (x,y) points for perspective transform calibration
    """
    vision_system.calibrate_from_points(corner_points)


def get_vision_parameters() -> Dict:
    """Get current vision system parameters"""
    return vision_system.params.copy()


def set_vision_parameters(new_params: Dict):
    """Update vision system parameters"""
    vision_system.params.update(new_params)
    vision_system.save_calibration()


def get_vision_presets() -> Dict:
    """Get available vision parameter presets"""
    return vision_system.get_presets()


def save_vision_preset(name: str, preset_params: Optional[Dict] = None) -> bool:
    """Save current or specified parameters as a preset"""
    return vision_system.save_preset(name, preset_params)


def load_vision_preset(name: str) -> bool:
    """Load a vision parameter preset"""
    return vision_system.load_preset(name)


def delete_vision_preset(name: str) -> bool:
    """Delete a vision parameter preset"""
    return vision_system.delete_preset(name)
