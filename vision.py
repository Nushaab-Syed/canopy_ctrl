import cv2

def process_frame(frame):
    """
    Input:
        frame: raw BGR frame from the camera
    Output:
        processed_frame: frame with overlays drawn
        result: dictionary with robot-relevant data (e.g. dx)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Convert back to BGR for visualization
    vis_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Placeholder value for now
    result = {
        "dx": 0.0,
        "alignment_ok": True
    }

    return vis_frame, result
