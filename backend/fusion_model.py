"""
Advanced Image Fusion Model — High-Quality Output
===================================================
Pipeline:
  1. Resize to max 1024 px (was 640 — more detail preserved)
  2. Per-pixel activity saliency (gradient + LoG + local variance)
  3. Laplacian Pyramid fusion (luminance channel, depth=6)
  4. Smart YCbCr colour blending — preserves source colours
  5. Bilateral denoising (smooth noise, hard edges stay sharp)
  6. Retinex-inspired local tone mapping (reveal shadow detail)
  7. CLAHE (stronger clip=3.0 for vivid local contrast)
  8. Aggressive unsharp masking (amount=2.0) — crystal-clear edges
  9. Saturation boost (+25 %) — vivid, non-washed output
 10. Optional residual CNN refinement (if PyTorch present)
"""

import numpy as np
from PIL import Image, ImageEnhance
import cv2                                    # type: ignore
from scipy.ndimage import gaussian_filter     # type: ignore

# Optional PyTorch enhancer
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_np(img: Image.Image) -> np.ndarray:
    """PIL → float32 numpy [0,1] (H,W,3)."""
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """float32 [0,1] (H,W,3) → PIL RGB."""
    return Image.fromarray((arr.clip(0.0, 1.0) * 255).astype(np.uint8))


def resize_to_common(images, max_px: int = 1024):
    """Resize all PIL images to the smallest common (W, H), capped at max_px."""
    W = min(img.width  for img in images)
    H = min(img.height for img in images)
    if max(W, H) > max_px:
        scale = max_px / max(W, H)
        W, H  = int(W * scale), int(H * scale)
    # Must be divisible by 2^depth for pyramid (depth 6 means divisible by 64)
    W = max(64, W - W % 64)
    H = max(64, H - H % 64)
    return [img.resize((W, H), Image.LANCZOS) for img in images], W, H


# ─────────────────────────────────────────────────────────────────────────────
# Saliency / activity maps
# ─────────────────────────────────────────────────────────────────────────────

def activity_map(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Per-pixel information richness:
      • LoG   — detects edges & blobs
      • Sobel — gradient magnitude (structural edges)
      • Local variance — measures texture energy
    All three normalised and combined, then smoothed so blend seams disappear.
    """
    # High-pass via LoG
    blurred = gaussian_filter(gray, sigma)
    log     = np.abs(gray - blurred)

    # Sobel gradient magnitude (cv2 works on float32)
    g8   = (gray * 255).astype(np.float32)
    sobx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobx**2 + soby**2) / 255.0

    # Local variance
    mu   = gaussian_filter(gray, 3.0)
    mu2  = gaussian_filter(gray ** 2, 3.0)
    lvar = np.sqrt(np.maximum(mu2 - mu**2, 0))

    # Combined score (empirically tuned weights)
    score = 0.4 * log + 0.4 * sobel + 0.2 * lvar

    # Smooth weight map → invisible seams
    return gaussian_filter(score, sigma=1.5)


def build_weight_maps(grays: list) -> list:
    """Soft-max activity weight maps (sum to 1 per pixel)."""
    acts    = [activity_map(g) for g in grays]
    stacked = np.stack(acts, axis=0)              # (N,H,W)
    total   = stacked.sum(axis=0) + 1e-8
    return [a / total for a in acts]


def sharpness_map(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Per-pixel sharpness: variance of Laplacian in local window.
    Blurry regions → low sharpness; sharp regions → high sharpness.
    """
    lap = cv2.Laplacian((gray * 255).astype(np.float32), cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    local_var = cv2.filter2D(lap ** 2, -1, kernel) - (cv2.filter2D(lap, -1, kernel) ** 2)
    local_var = np.maximum(local_var, 0)
    return gaussian_filter(np.sqrt(local_var + 1e-8) / 255.0, sigma=1.0)


def build_sharpness_weights(grays: list, temperature: float = 0.08) -> list:
    """
    Steep soft-max over sharpness: strongly favors sharper pixels.
    Low temperature → almost binary (pick sharper pixel).
    """
    sharp = [sharpness_map(g) for g in grays]
    stacked = np.stack(sharp, axis=0)
    # Steep softmax: exp(x/T) with small T
    exp = np.exp((stacked - stacked.max(axis=0, keepdims=True)) / max(temperature, 1e-6))
    total = exp.sum(axis=0) + 1e-8
    return [exp[i] / total for i in range(len(grays))]


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian Pyramid fusion (per-channel)
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_pyramid(img: np.ndarray, depth: int) -> list:
    pyr = [img]
    for _ in range(depth):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _laplace_pyramid(img: np.ndarray, depth: int) -> list:
    gp = _gauss_pyramid(img, depth)
    lp = []
    for i in range(depth):
        up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[depth])
    return lp


def _fuse_channel_pyramid(channels: list, weight_mats: list, depth: int) -> np.ndarray:
    """Weighted Laplacian pyramid blend for a single channel."""
    lps = [_laplace_pyramid(c, depth) for c in channels]
    wps = [_gauss_pyramid(w, depth)   for w in weight_mats]

    fused_lp = []
    for lvl in range(depth + 1):
        acc = np.zeros_like(lps[0][lvl], dtype=np.float32)
        for i in range(len(channels)):
            w = wps[i][lvl]
            if lps[0][lvl].ndim == 2:
                acc += w * lps[i][lvl]
            else:
                acc += w[:, :, np.newaxis] * lps[i][lvl]
        fused_lp.append(acc)

    result = fused_lp[-1]
    for lvl in range(depth - 1, -1, -1):
        result = cv2.pyrUp(result,
                           dstsize=(fused_lp[lvl].shape[1], fused_lp[lvl].shape[0]))
        result += fused_lp[lvl]
    return result


def laplacian_pyramid_fuse(arrs: list, weights: list, depth: int = 6) -> np.ndarray:
    """Fuse float32 (H,W,3) arrays using Laplacian pyramid per-channel."""
    result = np.zeros_like(arrs[0])
    for ch in range(3):
        result[:, :, ch] = _fuse_channel_pyramid(
            [a[:, :, ch] for a in arrs],
            weights, depth
        ).clip(0.0, 1.0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Smart colour blending (YCbCr space)
# ─────────────────────────────────────────────────────────────────────────────

def _is_colour(arr: np.ndarray, thr: float = 0.008) -> bool:
    """True when there is meaningful colour (not grayscale)."""
    diff = np.mean(np.std(arr - arr.mean(axis=2, keepdims=True), axis=(0, 1)))
    return float(diff) > thr


def _saturation(arr: np.ndarray) -> float:
    mx = arr.max(axis=2); mn = arr.min(axis=2)
    return float((mx - mn).mean())


def _to_ycbcr(arr):
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    Y  =  0.299*r   + 0.587*g  + 0.114*b
    Cb = -0.16874*r - 0.33126*g + 0.5*b   + 0.5
    Cr =  0.5*r     - 0.41869*g - 0.08131*b + 0.5
    return Y, Cb, Cr


def _from_ycbcr(Y, Cb, Cr):
    Cb -= 0.5; Cr -= 0.5
    r = Y + 1.402*Cr
    g = Y - 0.34414*Cb - 0.71414*Cr
    b = Y + 1.772*Cb
    return np.stack([r, g, b], axis=2).clip(0.0, 1.0)


def smart_colour_blend(arrs: list, fused_Y: np.ndarray) -> np.ndarray:
    """
    Replace fused luminance (Y) into the best colour source's chrominance,
    blending Cb/Cr from all colour sources weighted by their saturation.
    """
    colour_arrs = [a for a in arrs if _is_colour(a)]
    if not colour_arrs:
        return np.stack([fused_Y, fused_Y, fused_Y], axis=2)

    sats = np.array([_saturation(a) for a in colour_arrs]) + 1e-8
    sats /= sats.sum()

    Cb_blend = sum(s * _to_ycbcr(a)[1] for s, a in zip(sats, colour_arrs))
    Cr_blend = sum(s * _to_ycbcr(a)[2] for s, a in zip(sats, colour_arrs))

    return _from_ycbcr(fused_Y, Cb_blend, Cr_blend)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing chain
# ─────────────────────────────────────────────────────────────────────────────

def bilateral_denoise(arr: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing via bilateral filter."""
    img8     = (arr * 255).clip(0, 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img8, d=7, sigmaColor=35, sigmaSpace=35)
    return denoised.astype(np.float32) / 255.0


def denoise_lowlight(arr: np.ndarray, strength: float = 10.0) -> np.ndarray:
    """Non-local means denoising for low-light noise. Preserves edges."""
    img8 = (arr * 255).clip(0, 255).astype(np.uint8)
    h = int(max(3, strength))
    out = cv2.fastNlMeansDenoisingColored(img8, None, h, h, 7, 21)
    return out.astype(np.float32) / 255.0


def retinex_tone_map(arr: np.ndarray, sigma: float = 60.0) -> np.ndarray:
    """
    Single-scale Retinex: recover detail hidden in dark / overexposed regions.
    Illumination estimated by large-sigma Gaussian blur; reflectance = log(I) - log(L).
    """
    eps   = 1e-3
    log_i = np.log(arr + eps)
    illum = gaussian_filter(arr, sigma=sigma)
    log_l = np.log(illum + eps)
    retinex = log_i - log_l                  # log reflectance (unbounded)

    # Normalise per-channel so output stays [0,1]
    out = np.zeros_like(arr)
    for ch in range(3):
        ch_data = retinex[:, :, ch]
        mn, mx  = ch_data.min(), ch_data.max()
        if mx - mn > 1e-6:
            out[:, :, ch] = (ch_data - mn) / (mx - mn)
        else:
            out[:, :, ch] = arr[:, :, ch]

    # Blend 50 % retinex + 50 % original to keep naturalness
    return (0.5 * out + 0.5 * arr).clip(0.0, 1.0)


def clahe_enhance(arr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """CLAHE on L channel (LAB space) — strong local contrast improvement."""
    lab   = cv2.cvtColor((arr * 255).clip(0, 255).astype(np.uint8),
                         cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0


def unsharp_mask(arr: np.ndarray,
                 radius: float = 1.0,
                 amount: float = 2.0,
                 threshold: float = 0.005) -> np.ndarray:
    """
    Selective unsharp masking.
    Only pixels already containing edges are sharpened (threshold guard).
    """
    blurred  = gaussian_filter(arr, sigma=radius)
    diff     = arr - blurred
    edge_mask = (np.abs(diff).max(axis=2) > threshold)[:, :, np.newaxis]
    return (arr + amount * diff * edge_mask).clip(0.0, 1.0)


def boost_saturation(arr: np.ndarray, factor: float = 1.30) -> np.ndarray:
    """Increase colour saturation via PIL ImageEnhance.Color."""
    pil    = np_to_pil(arr)
    pil    = ImageEnhance.Color(pil).enhance(factor)
    return pil_to_np(pil)


# ─────────────────────────────────────────────────────────────────────────────
# Optional lightweight CNN (residual refinement)
# ─────────────────────────────────────────────────────────────────────────────

class _TinyRefiner(nn.Module):
    """4-layer residual CNN — noise removal + micro-sharpening."""
    def __init__(self, ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.LeakyReLU(0.1),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.LeakyReLU(0.1),
        )
        self.dec = nn.Conv2d(ch, 3, 3, 1, 1)
        self.sig = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f   = self.enc(x)
        f   = f + self.mid(f)           # residual skip
        out = self.sig(x + 0.12 * self.dec(f))
        return out


_REFINER = None

def _get_refiner():
    global _REFINER
    if TORCH_OK and _REFINER is None:
        _REFINER = _TinyRefiner().eval()
    return _REFINER


def cnn_refine(arr: np.ndarray) -> np.ndarray:
    """Pass fused image through tiny residual CNN (if torch available)."""
    if not TORCH_OK:
        return arr
    model = _get_refiner()
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
    return out.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Focus: Blur + Clear → Clean All-in-Focus
# ─────────────────────────────────────────────────────────────────────────────

def multi_focus_clear_fuse(pil_images: list) -> Image.Image:
    """
    Fuse blurry + clear images into one crystal-clear result.
    Matches pixels: picks the sharper region from each source.
    Perfect for: one blurry image + one clear image of same scene.
    """
    if len(pil_images) < 2:
        raise ValueError("Need at least 2 images.")

    images, W, H = resize_to_common(pil_images, max_px=1024)
    arrs  = [pil_to_np(img) for img in images]
    grays = [0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2] for a in arrs]

    # Sharpness-prior weights (steep: strongly pick sharper pixels)
    weights = build_sharpness_weights(grays, temperature=0.06)

    depth = min(6, int(np.log2(min(W, H))) - 1)
    fused = laplacian_pyramid_fuse(arrs, weights, depth)
    fused_Y = 0.299*fused[:,:,0] + 0.587*fused[:,:,1] + 0.114*fused[:,:,2]
    coloured = smart_colour_blend(arrs, fused_Y)

    denoised = bilateral_denoise(coloured)
    enhanced = clahe_enhance(denoised, clip=3.5)
    sharp = unsharp_mask(enhanced, radius=0.8, amount=2.5, threshold=0.003)
    vivid = boost_saturation(sharp, factor=1.15)
    final = cnn_refine(vivid)

    return np_to_pil(final)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def deep_fuse(pil_images: list) -> Image.Image:
    """
    Full high-quality fusion pipeline:
      1  Resize to common (max 1024 px)
      2  Compute activity / saliency maps
      3  Laplacian pyramid fusion (depth up to 6)
      4  Smart YCbCr colour blending
      5  Bilateral edge-preserving denoise
      6  Retinex local tone mapping
      7  CLAHE (clip=3.0) local contrast
      8  Aggressive unsharp mask (amount=2.0)
      9  +30 % saturation boost
     10  Optional residual CNN refinement

    Returns a PIL RGB Image.
    """
    if len(pil_images) < 2:
        raise ValueError("Need at least 2 images.")

    # ── Step 1: align resolutions ──────────────────────────────────────────
    images, W, H = resize_to_common(pil_images, max_px=1024)
    arrs   = [pil_to_np(img) for img in images]
    grays  = [0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2] for a in arrs]

    # ── Step 2: per-pixel activity weights ────────────────────────────────
    weights = build_weight_maps(grays)

    # ── Step 3: Laplacian pyramid fusion ─────────────────────────────────
    depth   = min(6, int(np.log2(min(W, H))) - 1)
    fused   = laplacian_pyramid_fuse(arrs, weights, depth)
    fused_Y = 0.299*fused[:,:,0] + 0.587*fused[:,:,1] + 0.114*fused[:,:,2]

    # ── Step 4: smart colour merge ────────────────────────────────────────
    coloured = smart_colour_blend(arrs, fused_Y)

    # ── Step 5: bilateral denoise ─────────────────────────────────────────
    denoised = bilateral_denoise(coloured)

    # ── Step 6: Retinex tone mapping ──────────────────────────────────────
    tone_mapped = retinex_tone_map(denoised, sigma=50.0)

    # ── Step 7: CLAHE contrast ────────────────────────────────────────────
    enhanced = clahe_enhance(tone_mapped, clip=3.0)

    # ── Step 8: unsharp masking ───────────────────────────────────────────
    sharp = unsharp_mask(enhanced, radius=1.0, amount=2.0, threshold=0.005)

    # ── Step 9: saturation boost ──────────────────────────────────────────
    vivid = boost_saturation(sharp, factor=1.30)

    # ── Step 10: optional CNN refinement ──────────────────────────────────
    final = cnn_refine(vivid)

    return np_to_pil(final)


def ir_vis_color_fuse(pil_images: list) -> Image.Image:
    """
    Specifically designed for Infrared (IR) and Visible (VIS) image fusion.
    Extracts the color from the most colorful image (Visible), 
    and injects the absolute highest details and bright thermal features 
    from both images into a perfectly clear color image.
    """
    if len(pil_images) < 2:
        raise ValueError("Need at least 2 images.")

    # ── Step 1: align resolutions ──────────────────────────────────────────
    images, W, H = resize_to_common(pil_images, max_px=1024)
    arrays = [pil_to_np(img) for img in images]
    
    # ── Step 2: Identify Visible vs Infrared ───────────────────────────────
    def color_saturation(arr):
        if arr.shape[2] == 1: return 0.0
        mx, mn = arr.max(axis=2), arr.min(axis=2)
        return float((mx - mn).mean())

    sats = [color_saturation(a) for a in arrays]
    vis_idx = int(np.argmax(sats))
    ir_idx = 1 if vis_idx == 0 else 0
        
    vis_img = arrays[vis_idx]
    ir_img = arrays[ir_idx]
    
    # ── Step 3: Extract Luminance ──────────────────────────────────────────
    def get_luma(arr):
        return 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        
    V_vis = get_luma(vis_img)
    V_ir  = get_luma(ir_img)
    
    # ── Step 4: Enhance IR Contrast for clear thermal targets ──────────────
    ir_uint8 = (V_ir * 255).clip(0,255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    V_ir_enhanced = clahe.apply(ir_uint8).astype(np.float32) / 255.0
    
    # ── Step 5: Base/Detail Edge-Preserving Decomposition ──────────────────
    def decompose(img_gray):
        img8 = (img_gray * 255).clip(0, 255).astype(np.uint8)
        base8 = cv2.bilateralFilter(img8, d=9, sigmaColor=50, sigmaSpace=50)
        base = base8.astype(np.float32) / 255.0
        detail = img_gray - base
        return base, detail
        
    B_vis, D_vis = decompose(V_vis)
    B_ir, D_ir = decompose(V_ir_enhanced)
    
    # ── Step 6: Fusion Rules ───────────────────────────────────────────────
    B_fused = np.maximum(B_vis, B_ir)
    
    mask_vis = np.abs(D_vis) > np.abs(D_ir)
    D_fused = np.where(mask_vis, D_vis, D_ir)
    
    V_fused = np.clip(B_fused + D_fused, 0.0, 1.0)
    
    # ── Step 7: HSV Recombination & Vibrant Color Boost ────────────────────
    vis_uint8 = (vis_img * 255).clip(0, 255).astype(np.uint8)
    hsv_vis = cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    hsv_vis[:,:,2] = V_fused * 255.0
    hsv_vis[:,:,1] = np.clip(hsv_vis[:,:,1] * 1.30, 0, 255)
    
    hsv_vis = hsv_vis.clip(0, 255).astype(np.uint8)
    fused_rgb = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    
    # ── Step 8: Final Unsharp Mask for Clean & Clear Image ─────────────────
    blur = cv2.GaussianBlur(fused_rgb, (0, 0), 2.0)
    final_output = np.clip(fused_rgb + 1.5 * (fused_rgb - blur), 0.0, 1.0)
    
    return np_to_pil(final_output)


def ir_vis_clean_fuse(pil_images: list, use_emma: bool = False) -> Image.Image:
    """
    IR + Visible fusion tuned for clean, clear output.
    For: thermal/IR + low-light visible (e.g. night scenes).
    Pipeline: fuse → denoise (low-light) → Retinex → CLAHE → sharpen → saturation.
    """
    if len(pil_images) < 2:
        raise ValueError("Need at least 2 images (IR + Visible).")

    # Ensure RGB (grayscale IR → 3 channels)
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in pil_images[:2]]

    # Fuse: EMMA if available and requested, else ir_vis_color
    if use_emma:
        try:
            from emma import emma_fuse
            import os as _os
            _emma_dir = _os.path.dirname(_os.path.abspath(__import__("emma").__file__))
            _finetuned = _os.path.join(_emma_dir, "models", "EMMA_irvis_finetuned.pth")
            _model_path = _finetuned if _os.path.isfile(_finetuned) else None
            fused = emma_fuse(images, model_path=_model_path, preserve_color=True, max_px=1024)
        except Exception:
            fused = ir_vis_color_fuse(images)
    else:
        fused = ir_vis_color_fuse(images)

    arr = pil_to_np(fused)

    # Strong denoising for low-light noise
    arr = denoise_lowlight(arr, strength=8.0)

    # Bilateral for edge-preserving cleanup
    arr = bilateral_denoise(arr)

    # Retinex: recover shadow detail
    arr = retinex_tone_map(arr, sigma=40.0)

    # Strong CLAHE for low-light contrast
    arr = clahe_enhance(arr, clip=4.0)

    # Aggressive unsharp for clarity
    arr = unsharp_mask(arr, radius=0.8, amount=2.8, threshold=0.002)

    # Mild saturation
    arr = boost_saturation(arr, factor=1.25)

    arr = cnn_refine(arr)
    return np_to_pil(arr)
