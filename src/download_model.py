# src/download_model.py
"""
Robust ArcFace ONNX model downloader with auto-retry and integrity verification.
Downloads 'w600k_r50.onnx' directly or via 'buffalo_l.zip'.
"""
import sys
import time
import urllib.request
import zipfile
import shutil
from pathlib import Path

MODEL_DIR = Path("models")
TARGET_ONNX = MODEL_DIR / "embedder_arcface.onnx"
ZIP_PATH = MODEL_DIR / "buffalo_l.zip"

# Primary & fallback URLs
DIRECT_ONNX_URLS = [
    "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    "https://huggingface.co/MonsterMMORPG/tools/resolve/main/w600k_r50.onnx",
]
ZIP_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download",
]


def download_stream(url: str, dest: Path, desc: str = "Downloading") -> bool:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.info().get("Content-Length", -1))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB

            with open(temp_dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r{desc}: {percent:3d}% [{mb_done:.1f} MB / {mb_tot:.1f} MB]")
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r{desc}: {mb_done:.1f} MB")
                    sys.stdout.flush()

            # Check if complete
            if total_size > 0 and downloaded < total_size:
                print(f"\n[!] Incomplete transfer ({downloaded}/{total_size} bytes). Retrying...")
                if temp_dest.exists():
                    temp_dest.unlink()
                return False

            if temp_dest.exists():
                if dest.exists():
                    dest.unlink()
                temp_dest.rename(dest)
            print("\n[+] Download finished!")
            return True

    except Exception as e:
        print(f"\n[!] Connection error on {url}: {e}")
        if temp_dest.exists():
            temp_dest.unlink()
        return False


def download_detector_models():
    """Download YuNet face detection ONNX model and Haar cascade XML if missing."""
    yunet_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    if not yunet_path.exists():
        print(f"\n[+] Downloading YuNet 5-point face detector model...")
        download_stream(yunet_url, yunet_path, desc="Downloading YuNet")

    haar_path = MODEL_DIR / "haarcascade_frontalface_default.xml"
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
    if not haar_path.exists():
        print(f"\n[+] Downloading Haar cascade face detector fallback...")
        download_stream(haar_url, haar_path, desc="Downloading Haar XML")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_detector_models()

    if TARGET_ONNX.exists() and TARGET_ONNX.stat().st_size > 100 * 1024 * 1024:
        print(f"[+] ArcFace ONNX model already installed and verified at: {TARGET_ONNX.resolve()}")
        return

    # Clean any corrupt previous zip
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("=======================================================")
    print("  Downloading ArcFace ONNX (w600k_r50.onnx)")
    print("=======================================================")

    # Attempt 1: Direct ONNX download (faster - no zip overhead, ~170 MB)
    for url in DIRECT_ONNX_URLS:
        print(f"\nAttempting direct model download from HuggingFace...")
        if download_stream(url, TARGET_ONNX, desc="Downloading ArcFace ONNX"):
            if TARGET_ONNX.exists() and TARGET_ONNX.stat().st_size > 50 * 1024 * 1024:
                print(f"[+] Successfully installed model to: {TARGET_ONNX}")
                return

    # Attempt 2: Zip package fallback
    for url in ZIP_URLS:
        print(f"\nAttempting download of buffalo_l.zip...")
        if download_stream(url, ZIP_PATH, desc="Downloading zip package"):
            try:
                print("Extracting 'w600k_r50.onnx' from zip archive...")
                with zipfile.ZipFile(str(ZIP_PATH), "r") as zip_ref:
                    zip_ref.extract("w600k_r50.onnx", path=str(MODEL_DIR))

                extracted = MODEL_DIR / "w600k_r50.onnx"
                if extracted.exists():
                    if TARGET_ONNX.exists():
                        TARGET_ONNX.unlink()
                    extracted.rename(TARGET_ONNX)
                    print(f"[+] Successfully installed model to: {TARGET_ONNX}")
                    if ZIP_PATH.exists():
                        ZIP_PATH.unlink()
                    return
            except zipfile.BadZipFile:
                print("[!] Corrupt zip file. Removing...")
                if ZIP_PATH.exists():
                    ZIP_PATH.unlink()

    print("\n" + "=" * 60)
    print("[!] Automated download couldn't complete due to network timeout.")
    print("Please download directly via browser from one of these links:")
    print("  1) https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx")
    print(f"  Save the downloaded file directly as: {TARGET_ONNX.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
