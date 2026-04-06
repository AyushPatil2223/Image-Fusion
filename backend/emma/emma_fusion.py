"""
EMMA (CVPR 2024) — Equivariant Multi-Modality Image Fusion
===========================================================
Pretrained model integration for clean, high-quality IR+Visible fusion.
Outputs crystal-clear fused images with preserved color and thermal detail.
"""

import os
import numpy as np
from PIL import Image
import cv2

try:
    import torch
    from .nets.Ufuser import Ufuser
    EMMA_AVAILABLE = True
except ImportError:
    EMMA_AVAILABLE = False
    Ufuser = None

# Model path relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "EMMA.pth")


def _color_saturation(arr: np.ndarray) -> float:
    """Measure color saturation of image (0 = grayscale)."""
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    return float((mx - mn).mean())


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to grayscale [0,1]."""
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(
        np.float32
    )


def _ensure_divisible_by_32(h: int, w: int) -> tuple:
    """Ensure dimensions are divisible by 32 for EMMA."""
    h1 = h - h % 32
    w1 = w - w % 32
    return max(32, h1), max(32, w1)


def _run_emma_single(model, ir: np.ndarray, vi: np.ndarray, device: str) -> np.ndarray:
    """Run EMMA on a single patch. IR, VI: (1,1,H,W) float32 [0,1]."""
    ir_t = torch.FloatTensor(ir).to(device)
    vi_t = torch.FloatTensor(vi).to(device)
    with torch.no_grad():
        out = model(ir_t, vi_t)
    fused = out.cpu().numpy().squeeze()
    return fused


def emma_fuse(
    pil_images: list,
    model_path: str = None,
    preserve_color: bool = True,
    max_px: int = 1024,
) -> Image.Image:
    """
    Fuse 2 images using EMMA pretrained model (CVPR 2024).
    Best for Infrared + Visible pairs. Produces clean, clear output.

    Args:
        pil_images: List of 2 PIL Images (IR and Visible, order detected automatically)
        model_path: Path to EMMA.pth. Uses default if None.
        preserve_color: If True, inject visible color into grayscale fusion (HSV).
        max_px: Max dimension for processing (memory/speed tradeoff).

    Returns:
        PIL RGB Image (clean, high-quality fusion).
    """
    if not EMMA_AVAILABLE:
        raise RuntimeError(
            "EMMA requires PyTorch. Install: pip install torch einops"
        )

    if len(pil_images) < 2:
        raise ValueError("EMMA requires exactly 2 images (IR and Visible).")

    model_path = model_path or DEFAULT_MODEL_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"EMMA model not found at {model_path}. "
            "Run: python -m emma.download_model"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ufuser().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Resize to common size
    W = min(img.width for img in pil_images)
    H = min(img.height for img in pil_images)
    if max(W, H) > max_px:
        scale = max_px / max(W, H)
        W, H = int(W * scale), int(H * scale)
    W = max(64, W - W % 32)
    H = max(64, H - H % 32)

    images = [img.resize((W, H), Image.LANCZOS) for img in pil_images[:2]]
    arrs = [np.array(img.convert("RGB"), dtype=np.float32) / 255.0 for img in images]

    # Identify IR (less color) vs Visible (more color)
    sats = [_color_saturation(a) for a in arrs]
    vis_idx = int(np.argmax(sats))
    ir_idx = 1 if vis_idx == 0 else 0

    ir_arr = _to_grayscale(arrs[ir_idx])
    vi_arr = _to_grayscale(arrs[vis_idx])
    vis_rgb = arrs[vis_idx]

    # Prepare tensors (1,1,H,W)
    ir_in = ir_arr[np.newaxis, np.newaxis, ...]
    vi_in = vi_arr[np.newaxis, np.newaxis, ...]

    h, w = ir_in.shape[2], ir_in.shape[3]
    h1, w1 = _ensure_divisible_by_32(h, w)

    if h1 == h and w1 == w:
        fused_gray = _run_emma_single(model, ir_in, vi_in, device)
    else:
        fused_temp = np.zeros((h, w), dtype=np.float32)
        # Upper-left
        ir_t = ir_in[:, :, :h1, :w1]
        vi_t = vi_in[:, :, :h1, :w1]
        f = _run_emma_single(model, ir_t, vi_t, device)
        fused_temp[:h1, :w1] = f

        h2, w2 = h % 32, w % 32
        if w1 != w:
            ir_t = ir_in[:, :, :h1, -w1:]
            vi_t = vi_in[:, :, :h1, -w1:]
            f = _run_emma_single(model, ir_t, vi_t, device)
            fused_temp[:h1, -w2:] = f[:, -w2:]

        if h1 != h:
            ir_t = ir_in[:, :, -h1:, :w1]
            vi_t = vi_in[:, :, -h1:, :w1]
            f = _run_emma_single(model, ir_t, vi_t, device)
            fused_temp[-h2:, :w1] = f[-h2:, :]

        if h1 != h and w1 != w:
            ir_t = ir_in[:, :, -h1:, -w1:]
            vi_t = vi_in[:, :, -h1:, -w1:]
            f = _run_emma_single(model, ir_t, vi_t, device)
            fused_temp[-h2:, -w2:] = f[-h2:, -w2:]

        fused_gray = fused_temp

    # Normalize to [0,1]
    mn, mx = fused_gray.min(), fused_gray.max()
    if mx - mn > 1e-8:
        fused_gray = (fused_gray - mn) / (mx - mn)
    else:
        fused_gray = np.clip(fused_gray, 0, 1)

    if preserve_color and vis_rgb.shape[2] == 3:
        # HSV: keep H,S from visible, use EMMA output as V
        vis_uint8 = (vis_rgb * 255).clip(0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        v_fused = (fused_gray * 255).clip(0, 255).astype(np.float32)
        hsv[:, :, 2] = v_fused
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        result_rgb = np.stack([fused_gray, fused_gray, fused_gray], axis=2)

    # Mild post-processing for clarity
    result_uint8 = (result_rgb * 255).clip(0, 255).astype(np.uint8)
    result_uint8 = cv2.bilateralFilter(result_uint8, 5, 25, 25)
    result = Image.fromarray(result_uint8)

    return result
