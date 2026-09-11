# src/embed.py
"""
ArcFace ONNX Embedder (CPU Execution Provider)
Extracts 512-dimensional L2-normalized face embeddings from aligned 112x112 images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class EmbeddingResult:
    embedding: np.ndarray  # (512,) float32, L2-normalized
    norm_before: float
    dim: int


class ArcFaceEmbedderONNX:
    """
    ArcFace ONNX Embedder (e.g. w600k_r50 or MobileFaceNet).
    Input: Aligned 112x112 BGR image.
    Output: 512-D L2-normalized vector.
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            print(f"[embed] ArcFace model not found at '{model_path}'. Attempting automated download...")
            try:
                from .download_model import main as download_model_main
                download_model_main()
            except Exception as e:
                print(f"[embed] Auto-download failed: {e}")

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ArcFace ONNX model not found at '{model_path}'.\n"
                    "Please run `python -m src.download_model` or place 'embedder_arcface.onnx' in the models/ directory."
                )

        # CPU Execution Provider
        self.sess = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        if debug:
            print("[embed] ArcFace ONNX Model loaded successfully.")
            print("[embed] Input shape:", self.sess.get_inputs()[0].shape)
            print("[embed] Output shape:", self.sess.get_outputs()[0].shape)

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Standard ArcFace normalization: (x - 127.5) / 128.0
        rgb = (rgb - 127.5) / 128.0
        # HWC to CHW and add batch dimension -> (1, 3, 112, 112)
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
        norm = float(np.linalg.norm(v) + eps)
        return (v / norm).astype(np.float32), norm

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        x = self._preprocess(aligned_bgr)
        outputs = self.sess.run([self.out_name], {self.in_name: x})
        v = outputs[0].reshape(-1).astype(np.float32)
        v_norm, norm_before = self._l2_normalize(v)
        return EmbeddingResult(v_norm, norm_before, v_norm.size)
