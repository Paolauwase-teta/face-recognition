# 5-Point Landmark Face Recognition & Tracking System

High-accuracy face recognition and tracking system using 5-point facial landmark alignment (Haar cascades / MediaPipe), ArcFace 512-D ONNX embeddings, and closed-loop servo tracking via ESP8266.

## Camera Configuration

The system automatically detects and prioritizes **physical external USB cameras** (e.g. `Wed Camera`) over the PC's built-in laptop camera (`Integrated Camera`) and virtual cameras (`EShare Virtual Camera`).

### List Connected Cameras
To see all connected cameras and which device is selected:
```bash
python -m src.camera --list-cams
```

Output:
```text
=================================================================
INDEX   | TYPE                 | DEVICE NAME
=================================================================
  0     | Internal PC Camera   | Integrated Camera
  1     | Virtual Camera       | EShare Virtual Camera
  2     | Physical External    | Wed Camera -> [SELECTED PHYSICAL]
=================================================================
```

### Automatic Physical Camera Usage
All entry points automatically select the physical camera by default (`--cam auto`):
- **Camera Test / Live HUD**: `python -m src.camera`
- **5-Point Face Detection**: `python -m src.detect`
- **Target Face Enrollment (Live Webcam)**: `python -m src.enroll --name Payola`
- **Target Face Enrollment (From Photos)**: `python -m src.enroll --name Payola --image-dir path/to/photos`
- **Target Tracking & Servo Base**: `python -m src.track_target --target Payola --port COM4`

### Manual Camera Override
If you want to manually specify a camera index or network stream:
```bash
# Explicit camera index (e.g. 2)
python -m src.track_target --target TargetName --cam 2

# IP / RTSP / HTTP Camera Stream
python -m src.track_target --target TargetName --cam http://192.168.1.100:8080/video
```
