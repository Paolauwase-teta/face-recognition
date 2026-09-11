# src/enroll.py
"""
Chapter 2: Face Enrollment
Captures multiple face samples, aligns them using 5-point landmarks,
extracts ArcFace ONNX embeddings, and stores them in data/db/face_db.npz.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Union, Optional
import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .camera import open_camera, print_camera_list

DATA_DIR = Path("data")
ENROLL_DIR = DATA_DIR / "enroll"
DB_DIR = DATA_DIR / "db"
DB_JSON = DB_DIR / "face_db.json"
DB_NPZ = DB_DIR / "face_db.npz"


def save_enrolled_identity(name: str, embeddings: list) -> bool:
    if len(embeddings) == 0:
        print("\n[!] No samples captured. Enrollment aborted.")
        return False

    # Compute template embedding: Mean vector + L2 normalization
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)

    # Load existing DB if present
    db_data = {}
    if DB_NPZ.exists():
        existing = np.load(DB_NPZ)
        db_data = {k: existing[k] for k in existing.files}

    db_data[name] = mean_embedding
    np.savez(DB_NPZ, **db_data)

    # Save metadata JSON
    meta = {
        "names": list(db_data.keys()),
        "embedding_dim": int(mean_embedding.size),
        "updated_at": time.time(),
    }
    with open(DB_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 50)
    print(f"[+] SUCCESSFULLY ENROLLED '{name}' ({len(embeddings)} samples)")
    print(f"Database saved to: {DB_NPZ}")
    print(f"Metadata saved to: {DB_JSON}")
    print("=" * 50)
    return True


def enroll_from_images(name: str, image_paths: list[Path]):
    person_enroll_dir = ENROLL_DIR / name
    person_enroll_dir.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"   ENROLLING TARGET IDENTITY FROM IMAGES: {name}")
    print(f"==========================================")

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80)
    embedder = ArcFaceEmbedderONNX()

    embeddings = []
    for img_path in image_paths:
        if not img_path.exists() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[!] Warning: Could not read {img_path}")
            continue

        faces = det.detect(frame, max_faces=1)
        if not faces:
            print(f"[-] No face detected in: {img_path.name}")
            continue

        f = faces[0]
        aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
        if aligned is not None and aligned.size:
            res = embedder.embed(aligned)
            embeddings.append(res.embedding)
            crop_path = person_enroll_dir / f"{img_path.stem}_aligned.jpg"
            cv2.imwrite(str(crop_path), aligned)
            print(f"[+] Processed: {img_path.name} -> enrolled sample #{len(embeddings)}")

    save_enrolled_identity(name, embeddings)


def enroll_identity(name: str, cam_index: Union[int, str] = "auto", num_samples: int = 15):
    person_enroll_dir = ENROLL_DIR / name
    person_enroll_dir.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"   ENROLLING TARGET IDENTITY: {name}")
    print(f"==========================================")
    print("Controls:")
    print("  SPACE / 'S' : Capture sample")
    print("  'A'         : Toggle auto-capture mode")
    print("  'Q'         : Save & Quit / Finish")
    print("------------------------------------------")

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80)
    embedder = ArcFaceEmbedderONNX()

    cap = open_camera(cam_index)

    embeddings = []
    saved_crops = []
    auto_mode = False
    last_auto_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=1)
        vis = frame.copy()
        aligned = None

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))

        # Status text
        status = f"Enrolling: {name} | Samples: {len(embeddings)}/{num_samples}"
        cv2.putText(vis, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"Auto: {'ON' if auto_mode else 'OFF'} (Press 'A' to toggle)",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Enrollment - Live View", vis)
        if aligned is not None and aligned.size:
            cv2.imshow("Enrollment - Aligned 112x112", cv2.resize(aligned, (224, 224)))

        key = cv2.waitKey(1) & 0xFF

        # Auto capture every 0.5s if face is well-aligned
        if auto_mode and aligned is not None and (time.time() - last_auto_time > 0.5):
            res = embedder.embed(aligned)
            embeddings.append(res.embedding)
            ts = int(time.time() * 1000)
            crop_path = person_enroll_dir / f"{ts}.jpg"
            cv2.imwrite(str(crop_path), aligned)
            saved_crops.append(crop_path)
            last_auto_time = time.time()
            print(f"[Enroll] Auto-captured sample {len(embeddings)}/{num_samples}")

        if key in (ord(" "), ord("s")) and aligned is not None:
            res = embedder.embed(aligned)
            embeddings.append(res.embedding)
            ts = int(time.time() * 1000)
            crop_path = person_enroll_dir / f"{ts}.jpg"
            cv2.imwrite(str(crop_path), aligned)
            saved_crops.append(crop_path)
            print(f"[Enroll] Captured sample {len(embeddings)}/{num_samples}")

        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"[Enroll] Auto mode: {'ENABLED' if auto_mode else 'DISABLED'}")

        elif key == ord("q") or len(embeddings) >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    save_enrolled_identity(name, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a new face identity")
    parser.add_argument("--name", required=False, default="Payola", help="Name of identity to enroll (default: 'Payola')")
    parser.add_argument("--cam", default="auto", help="Camera index, 'auto' for physical external camera, or stream URL")
    parser.add_argument("--samples", type=int, default=15, help="Number of face samples (default: 15)")
    parser.add_argument("--images", nargs="+", help="One or more image file paths to enroll from")
    parser.add_argument("--image-dir", help="Directory of photos (.jpg, .png) to enroll from")
    parser.add_argument("--list-cams", action="store_true", help="List detected cameras and exit")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_list()
    elif not args.name:
        parser.error("the following arguments are required: --name (unless --list-cams is specified)")
    elif args.image_dir:
        dir_path = Path(args.image_dir)
        files = [p for p in dir_path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        if not files:
            print(f"[!] No image files found in {dir_path}")
        else:
            enroll_from_images(args.name, sorted(files))
    elif args.images:
        files = [Path(p) for p in args.images]
        enroll_from_images(args.name, files)
    else:
        enroll_identity(args.name, args.cam, args.samples)
