import argparse
import time
import cv2
import numpy as np

from .camera import open_camera, print_camera_list
from .haar_5pt import Haar5ptDetector


def main():
    parser = argparse.ArgumentParser(description="5-Point Landmark Face Detection")
    parser.add_argument("--cam", default="auto", help="Camera index or 'auto' for physical external camera")
    parser.add_argument("--list-cams", action="store_true", help="List detected cameras and exit")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_list()
        return

    # Initialize 5-point Face Detector
    detector = Haar5ptDetector(debug=True)

    # Open physical camera by default
    camera = open_camera(args.cam)

    print(f"Face detection started (Backend: {detector.backend}).")
    print("Press Q to quit.")

    t_prev = time.time()
    fps = 0.0

    while True:
        success, frame = camera.read()
        if not success or frame is None:
            print("ERROR: Could not read frame.")
            break

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, now - t_prev))
        t_prev = now

        faces = detector.detect(frame, max_faces=5)

        vis = frame.copy()
        for face in faces:
            # Draw bounding box
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)

            # Draw 5 facial landmarks
            colors = [
                (255, 0, 0),    # Left eye - Blue
                (0, 0, 255),    # Right eye - Red
                (0, 255, 255),  # Nose - Yellow
                (255, 0, 255),  # Left mouth - Magenta
                (255, 255, 0),  # Right mouth - Cyan
            ]
            labels = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]
            for idx, (px, py) in enumerate(face.kps):
                pt = (int(px), int(py))
                color = colors[idx % len(colors)]
                cv2.circle(vis, pt, 4, color, -1)
                cv2.circle(vis, pt, 5, (255, 255, 255), 1)

        # Status text
        backend_str = f"Backend: {detector.backend} | Faces: {len(faces)} | FPS: {fps:.1f}"
        cv2.putText(vis, backend_str, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, "Press 'Q' to quit", (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("5-Point Landmark Face Detection", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")


if __name__ == "__main__":
    main()