
import os
import io
import uuid
import time
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import traceback
import cv2
from cv2 import dnn_superres

# Import fusion models
try:
    from fusion_model import deep_fuse, ir_vis_color_fuse, multi_focus_clear_fuse, ir_vis_clean_fuse
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"Fusion model could not be loaded: {e}")
    DEEP_LEARNING_AVAILABLE = False
    deep_fuse = ir_vis_color_fuse = multi_focus_clear_fuse = ir_vis_clean_fuse = None

# Import EMMA (CVPR 2024) pretrained model for clean IR+Visible fusion
try:
    from emma import emma_fuse, EMMA_AVAILABLE
except ImportError:
    EMMA_AVAILABLE = False
    emma_fuse = None

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def b64_to_pil(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def load_image_from_request(file_key: str, request_obj) -> Image.Image:
    """Load a PIL image from multipart upload or base64 JSON."""
    if file_key in request_obj.files:
        f = request_obj.files[file_key]
        return Image.open(f.stream).convert("RGB")
    data = request_obj.get_json(silent=True) or {}
    if file_key in data:
        return b64_to_pil(data[file_key]).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# Fusion algorithms (pure-NumPy, no torch required for demo)
# ---------------------------------------------------------------------------
def to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) float [0,1] → (H,W) float [0,1]."""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return arr
    return (arr - mn) / (mx - mn)


def fuse_average(imgs):
    stack = np.stack([np.array(i, dtype=np.float32) / 255.0 for i in imgs], axis=0)
    return normalize(stack.mean(axis=0))


def fuse_max(imgs):
    stack = np.stack([np.array(i, dtype=np.float32) / 255.0 for i in imgs], axis=0)
    return normalize(stack.max(axis=0))


def fuse_weighted_gradient(imgs):
    """
    Gradient-based weights: pixels with higher local gradient contribute more.
    """
    arrays = [np.array(i, dtype=np.float32) / 255.0 for i in imgs]
    grays  = [to_gray(a) for a in arrays]

    def gradient_mag(g):
        gx = np.gradient(g, axis=1)
        gy = np.gradient(g, axis=0)
        return np.sqrt(gx**2 + gy**2)

    grad_maps = [gradient_mag(g) for g in grays]
    grad_sum  = np.stack(grad_maps, axis=0).sum(axis=0) + 1e-8

    result = np.zeros_like(arrays[0])
    for arr, gm in zip(arrays, grad_maps):
        w = gm[:, :, np.newaxis] / grad_sum[:, :, np.newaxis]
        result += w * arr
    return normalize(result)


def fuse_laplacian_pyramid(imgs, levels=4):
    """Laplacian pyramid fusion — a classic multi-scale approach."""
    def build_gaussian_pyramid(img, levels):
        gp = [img.copy()]
        for _ in range(levels - 1):
            img = (img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]) / 4
            gp.append(img)
        return gp

    def upsample(img, ref_shape):
        from PIL import Image as PILImage
        h, w = ref_shape[:2]
        pil = PILImage.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        pil = pil.resize((w, h), PILImage.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0

    def build_laplacian_pyramid(img, levels):
        gp = build_gaussian_pyramid(img, levels)
        lp = []
        for i in range(levels - 1):
            up = upsample(gp[i + 1], gp[i].shape)
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    arrays = [np.array(i, dtype=np.float32) / 255.0 for i in imgs]
    pys    = [build_laplacian_pyramid(a, levels) for a in arrays]
    grays  = [to_gray(a) for a in arrays]

    def grad_at_level(arr, level):
        for _ in range(level):
            arr = arr[::2, ::2]
        gx = np.gradient(arr, axis=1)
        gy = np.gradient(arr, axis=0)
        return np.sqrt(gx**2 + gy**2) + 1e-8

    fused_py = []
    for lvl in range(levels):
        gmaps = [grad_at_level(g, lvl) for g in grays]
        gsum  = np.stack(gmaps, axis=0).sum(axis=0)
        fused_lvl = np.zeros_like(pys[0][lvl])
        for py, gm in zip(pys, gmaps):
            w = gm[:, :, np.newaxis] / gsum[:, :, np.newaxis]
            fused_lvl += w * py[lvl]
        fused_py.append(fused_lvl)

    # Reconstruct
    result = fused_py[-1]
    for i in range(levels - 2, -1, -1):
        result = upsample(result, fused_py[i].shape) + fused_py[i]
    return normalize(result)


def fuse_deep_learning(images):
    """
    Apply the PyTorch Deep Learning fusion model.
    Returns a float32 numpy array [0,1] shaped (H,W,3) — same contract as all other fuse_* functions.
    """
    if not DEEP_LEARNING_AVAILABLE:
        raise RuntimeError("Deep learning model is not available (PyTorch missing or model file error).")
    pil_result = deep_fuse(images)
    return np.array(pil_result, dtype=np.float32) / 255.0


def fuse_ir_vis_clean(images):
    """
    IR + Visible → Clean & Clear. For thermal + low-light visible (night scenes).
    Strong denoising, contrast, sharpening for crystal-clear output.
    """
    if ir_vis_clean_fuse is None:
        raise RuntimeError("Fusion model not available.")
    pil_result = ir_vis_clean_fuse(images, use_emma=EMMA_AVAILABLE)
    return np.array(pil_result, dtype=np.float32) / 255.0


def fuse_ir_vis_color(images):
    """
    State-of-the-art dual-scale HSV fusion designed for IR and VIS.
    Preserves vibrant visible colors while injecting sharp glowing infrared thermal details.
    """
    if not DEEP_LEARNING_AVAILABLE:
        raise RuntimeError("Fusion model is not available.")
    pil_result = ir_vis_color_fuse(images)
    return np.array(pil_result, dtype=np.float32) / 255.0


def fuse_multi_focus_clear(images):
    """
    Blur + Clear fusion: matches pixels, picks sharper regions, outputs clean all-in-focus.
    """
    if multi_focus_clear_fuse is None:
        raise RuntimeError("Fusion model not available.")
    pil_result = multi_focus_clear_fuse(images)
    return np.array(pil_result, dtype=np.float32) / 255.0


def fuse_emma(images):
    """
    EMMA (CVPR 2024) pretrained model — clean, crystal-clear IR+Visible fusion.
    Equivariant multi-modality fusion for state-of-the-art quality.
    """
    if not EMMA_AVAILABLE or emma_fuse is None:
        raise RuntimeError(
            "EMMA model is not available. Install: pip install torch einops, then run: python -m emma.download_model"
        )
    pil_result = emma_fuse(images, preserve_color=True)
    return np.array(pil_result, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Super Resolution (OpenCV)
# ---------------------------------------------------------------------------
class SuperResolutionEnhancer:
    def __init__(self, model_name='edsr', scale=4):
        self.sr = dnn_superres.DnnSuperResImpl_create()
        
        # Paths for models
        # Searching in root and SuperResolution folder
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_paths = [
            os.path.join(root_dir, "EDSR_x4.pb"),
            os.path.join(root_dir, "SuperResolution", "EDSR_x4.pb"),
            os.path.join(root_dir, "LapSRN_x8.pb"),
            os.path.join(root_dir, "SuperResolution", "LapSRN_x8.pb"),
        ]
        
        selected_path = None
        if model_name.lower() == 'edsr':
            for p in model_paths:
                if "EDSR_x4.pb" in p and os.path.exists(p):
                    selected_path = p
                    break
        elif model_name.lower() == 'lapsrn':
            for p in model_paths:
                if "LapSRN_x8.pb" in p and os.path.exists(p):
                    selected_path = p
                    break
                    
        if selected_path:
            self.sr.readModel(selected_path)
            self.sr.setModel(model_name.lower(), scale)
            self.available = True
        else:
            print(f"Super Resolution model {model_name} not found!")
            self.available = False

    def enhance(self, image_pil):
        if not self.available:
            return image_pil
        
        img_np = np.array(image_pil.convert("RGB"))
        # OpenCV uses BGR
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        upscaled_cv = self.sr.upsample(img_cv)
        
        # Back to RGB
        upscaled_rgb = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(upscaled_rgb)

# Global instances
try:
    SR_EDSR = SuperResolutionEnhancer('edsr', 4)
    SR_LAPSRN = SuperResolutionEnhancer('lapsrn', 8)
except Exception as e:
    print(f"Error initializing SR models: {e}")
    SR_EDSR = SR_LAPSRN = None


FUSION_METHODS = {
    "average":              fuse_average,
    "max":                  fuse_max,
    "gradient_weighted":    fuse_weighted_gradient,
    "laplacian_pyramid":    fuse_laplacian_pyramid,
    "multi_focus_clear":    fuse_multi_focus_clear,
    "ir_vis_clean":         fuse_ir_vis_clean,
    "deep_learning":        fuse_deep_learning,
    "ir_vis_color":         fuse_ir_vis_color,
}
if EMMA_AVAILABLE:
    FUSION_METHODS["emma"] = fuse_emma


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(fused: np.ndarray, sources: list) -> dict:
    from math import log2

    def entropy(arr):
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 1))
        hist = hist / (hist.sum() + 1e-8)
        return float(-np.sum(hist * np.log2(hist + 1e-8)))

    def ssim_simple(a, b):
        c1, c2 = 0.01**2, 0.03**2
        mu_a, mu_b = a.mean(), b.mean()
        sig_a  = a.var()
        sig_b  = b.var()
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()
        num    = (2*mu_a*mu_b + c1) * (2*sig_ab + c2)
        den    = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
        return float(num / (den + 1e-8))

    def mutual_info(a, b, bins=64):
        hist2d, _, _ = np.histogram2d(a.flatten(), b.flatten(), bins=bins, range=[[0,1],[0,1]])
        pxy   = hist2d / (hist2d.sum() + 1e-8)
        px    = pxy.sum(axis=1, keepdims=True) + 1e-8
        py    = pxy.sum(axis=0, keepdims=True) + 1e-8
        mi    = np.sum(pxy * np.log2(pxy / (px * py) + 1e-8))
        return float(mi)

    fused_g = to_gray(fused)
    source_grays = [to_gray(np.array(s, dtype=np.float32)/255.0) for s in sources]

    ssim_scores = [ssim_simple(fused_g, sg) for sg in source_grays]
    mi_scores   = [mutual_info(fused_g, sg) for sg in source_grays]

    return {
        "entropy":         round(entropy(fused_g), 4),
        "ssim_avg":        round(float(np.mean(ssim_scores)), 4),
        "ssim_per_source": [round(s, 4) for s in ssim_scores],
        "mi_avg":          round(float(np.mean(mi_scores)), 4),
        "mi_per_source":   [round(m, 4) for m in mi_scores],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Image Fusion API is running"})


@app.route("/api/methods", methods=["GET"])
def get_methods():
    methods = [
        {
            "id": "average",
            "name": "Average Fusion",
            "description": "Pixel-wise averaging of all source images. Fast and simple baseline.",
            "speed": "Fast",
            "quality": "Basic",
        },
        {
            "id": "max",
            "name": "Max Fusion",
            "description": "Takes the maximum intensity at each pixel across all inputs. Good for highlighting bright features.",
            "speed": "Fast",
            "quality": "Good",
        },
        {
            "id": "gradient_weighted",
            "name": "Gradient-Weighted Fusion",
            "description": "Assigns higher weights to pixels with stronger local gradients, preserving fine structural details.",
            "speed": "Medium",
            "quality": "Great",
        },
        {
            "id": "laplacian_pyramid",
            "name": "Laplacian Pyramid Fusion",
            "description": "Multi-scale Laplacian pyramid approach. Classic method to fuse complementary frequency bands from each image.",
            "speed": "Slow",
            "quality": "Excellent",
        },
        {
            "id": "multi_focus_clear",
            "name": "Blur+Clear → Clean (Multi-Focus)",
            "description": "Fuse blurry + clear images. Matches pixels, picks sharper regions. Output: crystal-clear all-in-focus.",
            "speed": "Medium",
            "quality": "Excellent",
        },
        {
            "id": "ir_vis_clean",
            "name": "IR+Visible → Clean & Clear",
            "description": "Thermal/IR + low-light visible → denoise, enhance, sharpen. Best for night scenes.",
            "speed": "Medium",
            "quality": "Excellent",
        },
    ]
    if DEEP_LEARNING_AVAILABLE:
        methods.append({
            "id": "deep_learning",
            "name": "Deep Fusion (CDDFuse-inspired)",
            "description": "Multi-scale CNN with correlation-driven attention decomposition.",
            "speed": "Very Slow",
            "quality": "State-of-the-art",
        })
        methods.append({
            "id": "ir_vis_color",
            "name": "IR+VIS Color Fusion",
            "description": "State-of-the-art dual-scale HSV fusion. Perfectly preserves visible colors while injecting sharp glowing infrared thermal details.",
            "speed": "Medium",
            "quality": "Exceptional",
        })
        if EMMA_AVAILABLE:
            methods.append({
                "id": "emma",
                "name": "EMMA (CVPR 2024)",
                "description": "Pretrained equivariant multi-modality fusion. Clean, crystal-clear output — best for IR+Visible pairs.",
                "speed": "Medium",
                "quality": "State-of-the-art",
            })
    return jsonify({"methods": methods})


@app.route("/api/fuse", methods=["POST"])
def fuse_images():
    try:
        start = time.time()

        # --- Read method ---
        method = request.form.get("method", "gradient_weighted")
        if method not in FUSION_METHODS:
            return jsonify({"error": f"Unknown method '{method}'"}), 400

        # --- Read images ---
        images = []
        file_keys = sorted([k for k in request.files if k.startswith("image")])
        for key in file_keys:
            f = request.files[key]
            img = Image.open(f.stream).convert("RGB")
            images.append(img)

        if len(images) < 2:
            return jsonify({"error": "Please upload at least 2 images"}), 400

        # Resize all to the smallest common size, capped at 1024px
        min_w = min(i.width  for i in images)
        min_h = min(i.height for i in images)
        
        max_px = 1024
        if max(min_w, min_h) > max_px:
            scale = max_px / max(min_w, min_h)
            min_w = int(min_w * scale)
            min_h = int(min_h * scale)
        
        # Ensure dimensions are nicely divisible by 64 to avoid NumPy broadcasting errors in pyramids
        min_w = max(64, min_w - (min_w % 64))
        min_h = max(64, min_h - (min_h % 64))
        
        images = [i.resize((min_w, min_h), Image.LANCZOS) for i in images]

        # --- Fuse ---
        fused_arr = FUSION_METHODS[method](images)
        fused_uint8 = (fused_arr * 255).clip(0, 255).astype(np.uint8)
        fused_img = Image.fromarray(fused_uint8)

        # --- Save result ---
        result_id = str(uuid.uuid4())[:8]
        result_filename = f"fused_{result_id}.png"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        fused_img.save(result_path)

        # --- Metrics ---
        metrics = compute_metrics(fused_arr, images)

        elapsed = round(time.time() - start, 3)

        return jsonify({
            "success":     True,
            "result_id":   result_id,
            "image_b64":   pil_to_b64(fused_img),
            "metrics":     metrics,
            "method":      method,
            "num_images":  len(images),
            "time_seconds": elapsed,
            "size":        {"width": min_w, "height": min_h},
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<filename>", methods=["GET"])
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


@app.route("/api/compare", methods=["POST"])
def compare_methods():
    """Run all fusion methods on the uploaded images and return all results."""
    try:
        images = []
        file_keys = sorted([k for k in request.files if k.startswith("image")])
        for key in file_keys:
            f = request.files[key]
            img = Image.open(f.stream).convert("RGB")
            images.append(img)

        if len(images) < 2:
            return jsonify({"error": "Need at least 2 images"}), 400

        # Resize all to the smallest common size, capped at 1024px
        min_w = min(i.width  for i in images)
        min_h = min(i.height for i in images)
        
        max_px = 1024
        if max(min_w, min_h) > max_px:
            scale = max_px / max(min_w, min_h)
            min_w = int(min_w * scale)
            min_h = int(min_h * scale)
        
        # Ensure dimensions are nicely divisible by 64 to avoid NumPy broadcasting errors in pyramids
        min_w = max(64, min_w - (min_w % 64))
        min_h = max(64, min_h - (min_h % 64))
        
        images = [i.resize((min_w, min_h), Image.LANCZOS) for i in images]

        results = {}
        for name, fn in FUSION_METHODS.items():
            t0 = time.time()
            arr = fn(images)
            elapsed = round(time.time() - t0, 3)
            uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
            img_out = Image.fromarray(uint8)
            metrics = compute_metrics(arr, images)
            results[name] = {
                "image_b64":    pil_to_b64(img_out),
                "metrics":      metrics,
                "time_seconds": elapsed,
            }

        return jsonify({"success": True, "results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/super-resolve", methods=["POST"])
def super_resolve():
    """Apply Super Resolution to a single image."""
    try:
        start = time.time()
        
        model_type = request.form.get("model", "edsr") # edsr or lapsrn
        
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
            
        f = request.files["image"]
        img = Image.open(f.stream).convert("RGB")
        
        # Decide which model to use
        if model_type == "lapsrn" and SR_LAPSRN and SR_LAPSRN.available:
            result_img = SR_LAPSRN.enhance(img)
            method_used = "LapSRN (8x)"
        elif SR_EDSR and SR_EDSR.available:
            result_img = SR_EDSR.enhance(img)
            method_used = "EDSR (4x)"
        else:
            return jsonify({"error": "Super Resolution model not available"}), 500
            
        # --- Save result ---
        result_id = str(uuid.uuid4())[:8]
        result_filename = f"sr_{result_id}.png"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result_img.save(result_path)
        
        elapsed = round(time.time() - start, 3)
        
        return jsonify({
            "success":      True,
            "result_id":    result_id,
            "image_b64":    pil_to_b64(result_img),
            "method":       method_used,
            "time_seconds": elapsed,
            "original_size": {"width": img.width, "height": img.height},
            "new_size":      {"width": result_img.width, "height": result_img.height},
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
