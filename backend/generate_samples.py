"""
Generate realistic synthetic test images for the Image Fusion Lab.

Pairs:
  1. Infrared-style + Visible-style (night scene)
  2. Multi-exposure (under / normal / over)
  3. MRI + CT (medical grayscale)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

OUT = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save(img, name):
    path = os.path.join(OUT, name)
    img.save(path)
    print(f"  Saved {name}")
    return path


def _make_sky(W, H):
    """Simple gradient sky."""
    arr = np.zeros((H, W, 3), dtype=np.float32)
    for y in range(H):
        t = y / H
        arr[y, :, 0] = (1 - t) * 15 + t * 30
        arr[y, :, 1] = (1 - t) * 15 + t * 45
        arr[y, :, 2] = (1 - t) * 35 + t * 80
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Pair 1: Infrared & Visible (urban night scene)
# ──────────────────────────────────────────────────────────────────────────────

def make_ir_visible_pair(W=480, H=360):
    print("\n[1/3] Infrared + Visible pair …")
    rng = np.random.default_rng(42)

    # ── Visible image (RGB, low light, slight noise) ──────────────────────────
    arr_v = _make_sky(W, H).astype(np.uint8)
    img_v = Image.fromarray(arr_v)
    d = ImageDraw.Draw(img_v)

    # Buildings silhouettes
    for bx in range(20, W - 20, 60):
        bh = rng.integers(80, 200)
        by = H - bh
        bw = rng.integers(40, 55)
        shade = int(rng.integers(20, 60))
        d.rectangle([bx, by, bx + bw, H], fill=(shade, shade, shade + 5))
        # windows
        for wy in range(by + 10, H - 10, 25):
            for wx in range(bx + 5, bx + bw - 5, 15):
                lit = rng.random() < 0.4
                c = (220, 200, 100) if lit else (10, 10, 20)
                d.rectangle([wx, wy, wx + 8, wy + 12], fill=c)

    # Streetlamps (bright spots)
    for lx in range(30, W - 30, 90):
        d.ellipse([lx - 15, H - 90 - 15, lx + 15, H - 90 + 15],
                  fill=(255, 230, 180))

    # Road
    d.rectangle([0, H - 60, W, H], fill=(25, 25, 25))
    # Road markings
    for mx in range(30, W - 30, 60):
        d.rectangle([mx, H - 40, mx + 30, H - 32], fill=(180, 180, 180))

    # Cars (visible)
    for cx in [80, 220, 360]:
        d.rectangle([cx, H - 60, cx + 60, H - 20], fill=(40, 40, 120))
        d.ellipse([cx + 5, H - 25, cx + 20, H - 10], fill=(20, 20, 20))
        d.ellipse([cx + 40, H - 25, cx + 55, H - 10], fill=(20, 20, 20))
        # headlights
        d.ellipse([cx + 55, H - 50, cx + 65, H - 35], fill=(255, 250, 200))

    img_v = img_v.filter(ImageFilter.GaussianBlur(0.4))
    # Add noise
    v_arr = np.array(img_v, dtype=np.float32)
    v_arr += rng.normal(0, 3, v_arr.shape)
    v_arr = v_arr.clip(0, 255).astype(np.uint8)
    img_v = Image.fromarray(v_arr)
    save(img_v, "visible_urban.png")

    # ── Infrared image (grayscale tones, heat signature style) ────────────────
    # IR: bright = hot (cars, people, lamps), background = cool
    arr_ir = np.zeros((H, W), dtype=np.float32)

    # Cool sky gradient
    for y in range(H):
        arr_ir[y, :] = 30 + 15 * (y / H)

    # Buildings are slightly warmer than sky
    for bx in range(20, W - 20, 60):
        bh_val = rng.integers(80, 200)
        by = H - bh_val
        bw = rng.integers(40, 55)
        w_real = min(bw, W - bx)
        arr_ir[by:H, bx:bx + w_real] = 60 + rng.random((H - by, w_real)) * 20

    # Road (warm asphalt)
    arr_ir[H - 60:H, :] = 80 + rng.random((60, W)) * 15

    # Cars = very hot engine blocks
    for cx in [80, 220, 360]:
        arr_ir[H - 60:H - 20, cx:cx + 60] = 200 + rng.random((40, 60)) * 40

    # Streetlamps = very bright
    for lx in range(30, W - 30, 90):
        for iy in range(max(0, H - 105), H - 75):
            for ix in range(max(0, lx - 15), min(W, lx + 15)):
                dist = ((iy - (H - 90)) ** 2 + (ix - lx) ** 2) ** 0.5
                arr_ir[iy, ix] = min(255, 250 - dist * 3)

    # People walking (heat blobs)
    for px, py in [(150, H - 70), (290, H - 65), (400, H - 68)]:
        for dy in range(-20, 20):
            for dx in range(-8, 8):
                if 0 <= py + dy < H and 0 <= px + dx < W:
                    dist = (dy ** 2 + dx ** 2) ** 0.5
                    val = 230 - dist * 4
                    if val > arr_ir[py + dy, px + dx]:
                        arr_ir[py + dy, px + dx] = val

    arr_ir += rng.normal(0, 2, arr_ir.shape)
    arr_ir = arr_ir.clip(0, 255).astype(np.uint8)

    # Apply thermal-style colormap (hot = yellow/white, cold = dark blue)
    ir_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    norm = arr_ir.astype(np.float32) / 255.0
    ir_rgb[:, :, 0] = (np.minimum(norm * 2, 1.0) * 255).astype(np.uint8)       # R
    ir_rgb[:, :, 1] = (np.maximum(norm * 2 - 1, 0) * 255).astype(np.uint8)     # G
    ir_rgb[:, :, 2] = (np.maximum(0.5 - norm, 0) * 2 * 255).astype(np.uint8)   # B

    img_ir = Image.fromarray(ir_rgb).filter(ImageFilter.GaussianBlur(0.5))
    save(img_ir, "infrared_urban.png")
    print("   → Use 'visible_urban.png' + 'infrared_urban.png' as source images.")


# ──────────────────────────────────────────────────────────────────────────────
# Pair 2: Multi-exposure (under / normal / over)
# ──────────────────────────────────────────────────────────────────────────────

def make_multi_exposure(W=480, H=360):
    print("\n[2/3] Multi-exposure triplet …")
    rng = np.random.default_rng(7)

    # Build a 'ground truth' HDR scene
    arr = np.zeros((H, W, 3), dtype=np.float32)

    # Sky gradient
    for y in range(H // 2):
        t = y / (H // 2)
        arr[y, :, 0] = 80 + t * 60
        arr[y, :, 1] = 140 + t * 50
        arr[y, :, 2] = 200 + t * 30

    # Sun disc (very bright)
    cy, cx = H // 4, 3 * W // 4
    for y in range(H):
        for x in range(W):
            dist = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
            glow = max(0.0, 1.0 - dist / 60)
            arr[y, x, 0] += glow * 200
            arr[y, x, 1] += glow * 180
            arr[y, x, 2] += glow * 80

    # Green hills
    for x in range(W):
        hill_h = int(H * 0.45 + 30 * np.sin(x * 0.03) + 20 * np.cos(x * 0.07))
        if 0 <= hill_h < H:
            arr[hill_h:, x, 0] = 30 + rng.random() * 20
            arr[hill_h:, x, 1] = 80 + rng.random() * 30
            arr[hill_h:, x, 2] = 25 + rng.random() * 15

    # Trees (simple dark vertical blobs)
    for tx in range(20, W - 20, 50):
        th = rng.integers(60, 130)
        ty = int(H * 0.45 + 30 * np.sin(tx * 0.03)) - th
        w2 = rng.integers(12, 22)
        for dy in range(th):
            for dx in range(-w2, w2 + 1):
                if 0 <= ty + dy < H and 0 <= tx + dx < W:
                    arr[ty + dy, tx + dx, 0] = 20 + rng.random() * 10
                    arr[ty + dy, tx + dx, 1] = 55 + rng.random() * 20
                    arr[ty + dy, tx + dx, 2] = 15 + rng.random() * 10

    arr = arr.clip(0, 255)
    base = Image.fromarray(arr.astype(np.uint8))

    def exposure(img, ev):
        factor = 2 ** ev
        e = ImageEnhance.Brightness(img)
        return e.enhance(factor)

    under = exposure(base, -1.8)
    normal = exposure(base, 0.0)
    over  = exposure(base, 1.8)

    save(under.filter(ImageFilter.GaussianBlur(0.3)),  "exposure_under.png")
    save(normal.filter(ImageFilter.GaussianBlur(0.3)), "exposure_normal.png")
    save(over.filter(ImageFilter.GaussianBlur(0.3)),   "exposure_over.png")
    print("   → Use all three 'exposure_*.png' as source images.")


# ──────────────────────────────────────────────────────────────────────────────
# Pair 3: Medical (MRI + CT brain slice)
# ──────────────────────────────────────────────────────────────────────────────

def make_medical_pair(W=256, H=256):
    print("\n[3/3] Medical MRI + CT pair …")
    rng = np.random.default_rng(13)

    def _brain_mask(W, H):
        mask = np.zeros((H, W), dtype=np.float32)
        cy, cx = H // 2, W // 2
        ry, rx = int(H * 0.42), int(W * 0.45)
        for y in range(H):
            for x in range(W):
                v = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
                mask[y, x] = 1.0 if v <= 1.0 else 0.0
        return mask

    mask = _brain_mask(W, H)

    # ─ MRI T1 (soft tissue detail, grey matter / white matter) ──────────────
    arr_mri = np.zeros((H, W), dtype=np.float32)
    # Background + skull
    skull_outer = _brain_mask(W, H)
    skull_inner = np.zeros((H, W), dtype=np.float32)
    cy2, cx2 = H // 2, W // 2
    for y in range(H):
        for x in range(W):
            v = ((y - cy2) / (H * 0.38)) ** 2 + ((x - cx2) / (W * 0.40)) ** 2
            skull_inner[y, x] = 1.0 if v <= 1.0 else 0.0

    skull = skull_outer - skull_inner
    arr_mri += skull * 180

    # White matter core
    wm = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            v = ((y - cy2) / (H * 0.28)) ** 2 + ((x - cx2) / (W * 0.30)) ** 2
            wm[y, x] = 1.0 if v <= 1.0 else 0.0
    arr_mri += wm * 220

    # Grey matter ring
    gm = skull_inner - wm
    arr_mri += gm * 160

    # Sulci / gyri pattern (subtle texture)
    x_coords = np.linspace(-np.pi, np.pi, W)
    y_coords = np.linspace(-np.pi, np.pi, H)
    XX, YY = np.meshgrid(x_coords, y_coords)
    pattern  = 15 * np.sin(3 * XX) * np.cos(4 * YY)
    pattern += 10 * np.sin(7 * XX + 2) * np.sin(5 * YY - 1)
    arr_mri += skull_inner * pattern

    # Ventricles (dark on MRI T1)
    for vcy, vcx, vry, vrx in [(cy2 - 20, cx2 - 15, 20, 12),
                                 (cy2 - 20, cx2 + 15, 20, 12)]:
        for y in range(H):
            for x in range(W):
                v = ((y - vcy) / vry) ** 2 + ((x - vcx) / vrx) ** 2
                if v <= 1.0:
                    arr_mri[y, x] = 10

    arr_mri += rng.normal(0, 4, arr_mri.shape)
    arr_mri = (arr_mri * mask).clip(0, 255)

    # Convert grayscale to 3-channel for the app
    mri_rgb = np.stack([arr_mri] * 3, axis=-1).astype(np.uint8)
    img_mri = Image.fromarray(mri_rgb).filter(ImageFilter.GaussianBlur(0.6))
    save(img_mri, "mri_brain.png")

    # ─ CT (bone bright, soft tissue moderate, CSF/air dark) ─────────────────
    arr_ct = np.zeros((H, W), dtype=np.float32)
    arr_ct += skull * 255            # bone = very bright on CT
    arr_ct += (gm + wm) * 80        # soft tissue medium
    # CSF in ventricles = dark
    for vcy, vcx, vry, vrx in [(cy2 - 20, cx2 - 15, 20, 12),
                                 (cy2 - 20, cx2 + 15, 20, 12)]:
        for y in range(H):
            for x in range(W):
                v = ((y - vcy) / vry) ** 2 + ((x - vcx) / vrx) ** 2
                if v <= 1.0:
                    arr_ct[y, x] = 5

    arr_ct += rng.normal(0, 6, arr_ct.shape)
    arr_ct = (arr_ct * mask).clip(0, 255)

    ct_rgb = np.stack([arr_ct] * 3, axis=-1).astype(np.uint8)
    img_ct = Image.fromarray(ct_rgb).filter(ImageFilter.GaussianBlur(0.5))
    save(img_ct, "ct_brain.png")

    print("   → Use 'mri_brain.png' + 'ct_brain.png' as source images.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Image Fusion Lab — Sample Image Generator")
    print("=" * 60)
    make_ir_visible_pair()
    make_multi_exposure()
    make_medical_pair()
    print(f"\nAll images saved to:  {OUT}/")
    print("\nSuggested combinations:")
    print("  • infrared_urban.png + visible_urban.png")
    print("  • exposure_under.png + exposure_normal.png + exposure_over.png")
    print("  • mri_brain.png + ct_brain.png")
    print("=" * 60)
