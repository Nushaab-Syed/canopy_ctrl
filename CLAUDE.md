# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an autonomous roofing robot system that places and nails shingles in rows using computer vision for precise alignment. The project is part of a construction automation startup participating in the University of Waterloo's Velocity Incubator Summer Accelerator program. The system combines a Raspberry Pi 5 for vision processing and control logic with an STM32 Nucleo for motor control, communicating via structured serial protocol.

## System Architecture

**Hardware Stack:**
- **Raspberry Pi 5**: Main computer running Flask web server, OpenCV vision processing, and FSM logic
- **STM32 Nucleo**: Motor controller handling stepper/servo control via structured serial protocol
- **USB Camera**: Downward-facing camera for shingle edge detection and alignment

**Software Components:**
- **app.py**: Main Flask application with robot state management, camera streaming, and API endpoints
- **vision.py**: Computer vision processing module for shingle edge detection and dx calculation using perspective transform
- **nucleo_comms.py**: Serial communication module for STM32 integration (to be implemented)
- **static/index.html**: Web dashboard for robot control and monitoring
- **camera_test.py**: Utility script for testing camera connectivity

**Control Flow:**
1. Camera captures downward-facing images of shingle placement area
2. Vision pipeline detects shingle edges and calculates alignment offset (dx)
3. FSM processes vision data and triggers appropriate placement states
4. Serial commands sent to STM32 for motor control (step values, control flags)
5. Web interface provides real-time monitoring and manual control

## Development Commands

```bash
# Run the application (requires virtual environment setup)
./venv/bin/python app.py

# Test camera connectivity
./venv/bin/python camera_test.py
```

## Dependencies

The project uses a Python virtual environment (`venv/`) and requires:
- Flask (web framework)
- OpenCV (cv2) for computer vision
- Flask-CORS for cross-origin requests

## API Endpoints

- `GET /status` - Returns current robot state (fsm_state, dx, shingle_count)
- `POST /command` - Accepts robot commands (start, place_shingle, stop)
- `GET /video_feed` - Streams processed camera feed
- `GET /` - Serves the web dashboard

## Vision Processing

The vision pipeline (`process_frame` in vision.py) converts camera frames to grayscale, applies Gaussian blur, performs edge detection with Canny, and returns processed frames with alignment data for the robot FSM. The camera is mounted on the moving bracket looking toward the ground to detect shingle edges and provide positioning feedback for accurate placement relative to previously installed shingles.

## Implementation Status & Roadmap

**✅ Currently Implemented:**
- Basic Flask web server with robot state management
- Camera streaming and basic edge detection
- Web dashboard with control buttons and status display

**🔧 To Implement:**
- **Computer Vision**: Shingle edge detection with dx calculation, perspective transform integration, calibration logic
- **FSM & Motion Control**: Complete placement state machine (aligning, placing, nailing), STM32 serial integration, movement synchronization
- **Communication**: nucleo_comms.py module, periodic status polling, motor feedback integration
- **Web Interface**: Processed frame overlay, motor status display, calibration/emergency controls
- **Deployment**: Headless boot scripts, network configuration, GitHub integration

## Project Context

This prototype demonstrates the startup's construction automation capabilities as part of the Velocity Incubator Summer Accelerator program. Led by a Mechatronics Engineering graduate from University of Waterloo, the project serves as technical validation before potentially pivoting to prefab housing automation. Emphasis on cross-domain skill development including AI/ML, embedded systems, backend development, and infrastructure.