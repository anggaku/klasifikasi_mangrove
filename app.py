import os, io, json, math, uuid, pickle, base64, random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import base64

import numpy as np
import cv2
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
import uuid

# ==== TF/Keras ====
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.applications import EfficientNetB0  # hanya jika nanti rebuild model
from keras.applications.efficientnet import preprocess_input as effnet_preprocess



# ==== fitur manual ====
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ------------------ konfigurasi umum ------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try: keras.utils.set_random_seed(SEED)
except Exception: pass

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EXPORT_DIR = STATIC_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
USE_MASKED_INPUT = True          # putihkan background biar tahan domain shift
MIN_AREA_RATIO   = 0.005         # 0.5% area kontur minimal

# --- lokasi artefak model ---
ARTIFACT_DIR = BASE_DIR / "artifactsv2"      # <- sesuai struktur kamu
CANDIDATE_MODELS = [
    ARTIFACT_DIR / "best_hybrid.keras",      # hybrid (.keras)
    ARTIFACT_DIR / "best_hybrid.h5",         # hybrid (h5)
    BASE_DIR / "best_model_compressed.h5",   # CNN-only (file kamu)
]

KEY_FEATURES = [
    "area","perimeter","form_factor","aspect_ratio","extent","solidity","eccentricity",
    "glcm_contrast","glcm_energy","glcm_homogeneity","h_mean","s_mean","v_mean"
]

# ------------------ loader fleksibel ------------------
def load_model_flex() -> keras.Model:
    last_err = None
    for p in CANDIDATE_MODELS:
        if p.exists():
            try:
                return keras.models.load_model(p)
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Tidak menemukan model yang valid. Dicoba: {CANDIDATE_MODELS}. Error terakhir: {last_err}")

def load_labeling_and_scaler():
    meta = scaler = le = None
    meta_p   = ARTIFACT_DIR / "meta.json"
    scaler_p = ARTIFACT_DIR / "scaler.pkl"
    le_p     = ARTIFACT_DIR / "label_encoder.pkl"
    if meta_p.exists() and scaler_p.exists() and le_p.exists():
        with open(meta_p) as f: meta = json.load(f)
        with open(scaler_p, "rb") as f: scaler = pickle.load(f)
        with open(le_p, "rb") as f: le = pickle.load(f)
    return meta, scaler, le

model = load_model_flex()
meta, scaler, le = load_labeling_and_scaler()

try:
    n_inputs = len(model.inputs)
except Exception:
    n_inputs = 1

IS_HYBRID = (meta is not None and scaler is not None and le is not None and n_inputs == 2)
FEATURE_COLS: List[str] = (meta.get("feature_cols", []) if meta else [])
CLASSES: List[str] = (meta.get("classes", []) if meta else [])
if not IS_HYBRID and not CLASSES:
    try:
        n = int(model.outputs[0].shape[-1])
        CLASSES = [f"class_{i}" for i in range(n)]
    except Exception:
        CLASSES = []
if meta and "img_size" in meta:
    IMG_SIZE = tuple(meta["img_size"])

def save_and_get_url(bgr: np.ndarray, filename_hint: str) -> str:
    name = filename_hint.replace(" ", "_")
    out_path = UPLOAD_DIR / name
    cv2.imwrite(str(out_path), bgr)
    return url_for("static", filename=f"uploads/{name}")

# ------------------ fitur manual (dipakai bila hybrid) ------------------
def segment_leaf_mask(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return np.zeros((1,1), np.uint8)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    _, th_s = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_v = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(th_s, th_v)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return mask
    c = cnts[int(np.argmax([cv2.contourArea(x) for x in cnts]))]
    out = np.zeros((h,w), dtype=np.uint8)
    cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    if cv2.contourArea(c) < MIN_AREA_RATIO * (h*w):
        out = mask
    return out

def shape_features(mask: np.ndarray) -> Dict[str, float]:
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return {k: 0.0 for k in [
            "area","perimeter","form_factor","aspect_ratio","extent",
            "solidity","eccentricity", *[f"hu_{i+1}" for i in range(7)]
        ]}
    c = cnts[int(np.argmax([cv2.contourArea(x) for x in cnts]))]
    area = cv2.contourArea(c)
    per  = cv2.arcLength(c, True)
    ff   = 4*np.pi*area/(per**2 + 1e-8)
    x,y,bw,bh = cv2.boundingRect(c)
    ar   = bw/(bh + 1e-8)
    extent = area/(bw*bh + 1e-8)
    hull = cv2.convexHull(c)
    ha   = cv2.contourArea(hull)
    solidity = area/(ha + 1e-8) if ha>0 else 0.0
    if len(c)>=5:
        (_, _),(MA,ma),_ = cv2.fitEllipse(c)
        a = max(MA,ma)/2.0; b = min(MA,ma)/2.0
        ecc = np.sqrt(max(a*a - b*b, 0.0))/(a + 1e-8) if a>0 else 0.0
    else:
        ecc = 0.0
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu_log = [(-1 if h<0 else 1)*np.log10(abs(h)+1e-30) for h in hu]
    out = {
        "area": float(area/(h*w)), "perimeter": float(per/(h+w)),
        "form_factor": float(ff), "aspect_ratio": float(ar),
        "extent": float(extent), "solidity": float(solidity), "eccentricity": float(ecc),
    }
    for i,v in enumerate(hu_log):
        out[f"hu_{i+1}"] = float(v)
    return out

def texture_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    gm = np.where(mask>0, gray, 0)
    gq = (gm/32).astype(np.uint8)
    glcm = graycomatrix(gq, distances=[1,2,4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    props = {}
    for prop in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
        props[prop] = float(np.mean(graycoprops(glcm, prop)))
    lbp = local_binary_pattern(gm, P=8, R=1, method="uniform")
    lbp_masked = lbp[mask>0]
    n_bins = 8+2
    hist, _ = np.histogram(lbp_masked, bins=np.arange(0, n_bins+1), density=True)
    out = {f"glcm_{k}": v for k,v in props.items()}
    out.update({f"lbp_{i}": float(hist[i]) for i in range(n_bins)})
    return out

def color_features(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    valid = mask>0
    if np.count_nonzero(valid)==0:
        return {"h_mean":0.0,"s_mean":0.0,"v_mean":0.0,"h_std":0.0,"s_std":0.0,"v_std":0.0}
    H = hsv[:,:,0][valid]; S = hsv[:,:,1][valid]; V = hsv[:,:,2][valid]
    return {
        "h_mean": float(np.mean(H)), "s_mean": float(np.mean(S)), "v_mean": float(np.mean(V)),
        "h_std": float(np.std(H)),  "s_std": float(np.std(S)),  "v_std": float(np.std(V)),
    }

def extract_manual_features_from_bgr(bgr: np.ndarray) -> Dict[str, float]:
    mask = segment_leaf_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    feats = {}
    feats.update(shape_features(mask))
    feats.update(texture_features(gray, mask))
    feats.update(color_features(bgr, mask))
    return feats

# ------------------ preprocess gambar ------------------
def mask_background_to_white(bgr: np.ndarray) -> np.ndarray:
    mask = segment_leaf_mask(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = 255
    return rgb

def preprocess_for_model(bgr: np.ndarray) -> np.ndarray:
    if USE_MASKED_INPUT:
        rgb = mask_background_to_white(bgr)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SIZE)
    img = img.astype(np.float32)
    img = effnet_preprocess(img)
    return img

def draw_annotated(bgr: np.ndarray, label: str, conf: float) -> np.ndarray:
    out = bgr.copy()
    txt = f"{label} ({conf:.2f})"
    # teks kecil + auto-scale
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.40, min(0.90, (min(h, w) / 800) * 0.6))
    thickness = max(1, int(scale * 2))
    (tw, th), base = cv2.getTextSize(txt, font, scale, thickness)
    x, y = 10, 10 + th
    cv2.putText(out, txt, (x, y), font, scale, (0,0,255), thickness, cv2.LINE_AA)
    return out

# ------------------ prediksi fleksibel ------------------
def predict_bgr(bgr: np.ndarray):
    X_img = np.expand_dims(preprocess_for_model(bgr), axis=0)
    if IS_HYBRID:
        feats = extract_manual_features_from_bgr(bgr)
        vec = np.array([[feats.get(c, 0.0) for c in FEATURE_COLS]], dtype=np.float32)
        vec = scaler.transform(vec)
        probs = model.predict([X_img, vec], verbose=0)[0]
    else:
        probs = model.predict(X_img, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    pred_label = (CLASSES[top_idx] if CLASSES else f"class_{top_idx}")
    order = np.argsort(probs)[::-1][:3]
    topk = [(CLASSES[i] if CLASSES else f"class_{i}", float(probs[i])) for i in order]
    return pred_label, top_conf, topk


def bgr_from_filestorage(file):
    # Pastikan pointer di awal sebelum dibaca
    try:
        file.stream.seek(0)
    except Exception:
        pass
    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Gagal membaca gambar (format tidak didukung).")
    return bgr

def to_data_uri(bgr: np.ndarray, quality: int = 85) -> str:
    """
    Konversi citra BGR (OpenCV) ke data URI (base64 JPEG) untuk langsung ditampilkan di <img src="...">.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Gambar kosong/None pada to_data_uri")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Gagal meng-encode gambar ke JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def draw_info_panel(img: np.ndarray, lines, topleft=(10, 10), alpha=0.35,
                    font_scale=0.5, thickness=1, color=(255,255,255)):
    x, y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    # hitung ukuran box
    line_h = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 6
    box_w = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in lines) if lines else 0
    box_h = line_h * len(lines)
    # overlay semi-transparan
    overlay = img.copy()
    cv2.rectangle(overlay, (x-6, y-6), (x + box_w + 6, y + box_h + 6), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # tulis teks
    for i, t in enumerate(lines):
        ty = y + i * line_h + (line_h - 6)
        cv2.putText(img, t, (x, ty), font, font_scale, color, thickness, cv2.LINE_AA)


def annotated_with_features(
    bgr: np.ndarray, 
    pred: str, 
    conf: float, 
    feats: Dict[str, float],
    draw_label_on_image: bool = True  # default: tidak tulis teks di gambar
) -> np.ndarray:
    vis = bgr.copy()

    # Segmentasi & kontur
    mask = segment_leaf_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return vis
    c = cnts[int(np.argmax([cv2.contourArea(x) for x in cnts]))]

    # Kontur (hijau)
    cv2.drawContours(vis, [c], -1, (0,255,0), 2)

    # Axis-aligned bbox (cyan) → aspect ratio
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis, (x,y), (x+w, y+h), (255,255,0), 2)

    # Rotated box (oranye)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(vis, [box], True, (0,165,255), 2)

    # Ellipse + sumbu mayor/minor
    if len(c) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
        cv2.ellipse(vis, ((int(cx), int(cy)), (int(MA), int(ma)), angle), (255,0,255), 2)
        a = MA/2.0; b = ma/2.0
        ang = math.radians(angle)
        dx, dy = math.cos(ang), math.sin(ang)
        # mayor (merah)
        p1 = (int(cx - a*dx), int(cy - a*dy))
        p2 = (int(cx + a*dx), int(cy + a*dy))
        cv2.line(vis, p1, p2, (0,0,255), 2)
        # minor (biru)
        pdx, pdy = -dy, dx
        p3 = (int(cx - b*pdx), int(cy - b*pdy))
        p4 = (int(cx + b*pdx), int(cy + b*pdy))
        cv2.line(vis, p3, p4, (255,0,0), 2)

    # Centroid (kuning)
    M = cv2.moments(c)
    if M["m00"] != 0:
        ccx, ccy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        cv2.circle(vis, (ccx, ccy), 4, (0,255,255), -1)

    # (opsional) label kecil di gambar
    if draw_label_on_image:
        vis = annotated_copy(vis, pred, conf, color=(0, 0, 0))

    return vis

def annotated_copy(bgr: np.ndarray, label: str, conf: float, color=(0, 0, 0)) -> np.ndarray:
    """Salin gambar BGR dan tulis teks kecil 'label (conf)' di kiri atas."""
    out = bgr.copy()
    txt = f"{label} ({conf:.2f})"
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.40, min(0.90, (min(h, w) / 800.0) * 0.6))  # kecil & adaptif
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
    x, y = 10, 10 + th
    cv2.putText(out, txt, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return out

# ------------------ Flask routes ------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    # Homepage sederhana
    return render_template("index.html",
                           classes=CLASSES, masked=USE_MASKED_INPUT, is_hybrid=IS_HYBRID)

@app.route("/deteksi", methods=["GET", "POST"])
def deteksi():
    results, error = [], None
    csv_relpath = None
    show_features = False

    if request.method == "POST":
        files = request.files.getlist("files")
        show_features = request.form.get("show_features") == "1"

        if not files or files[0].filename == "":
            error = "Pilih minimal satu gambar."
        else:
            rows = []  # untuk CSV
            for f in files:
                try:
                    
                    bgr = bgr_from_filestorage(f)
                    # ekstraksi fitur manual
                    feats = extract_manual_features_from_bgr(bgr)

                    # prediksi
                    pred, conf, topk = predict_bgr(bgr)

                    # anotasi untuk tampilan
                    # annotated_copy = draw_annotated
                    # (1) ekstrak fitur manual untuk panel
                    feats = extract_manual_features_from_bgr(bgr)

                    # (2) gambar overlay fitur
                    anno = annotated_with_features(bgr, pred, conf, feats, draw_label_on_image=True)
                    # (3) simpan ke static/uploads dan ambil URL
                    img_url = save_and_get_url(anno, f"anno_{uuid.uuid4().hex}.jpg")

                    # ringkasan fitur (ditampilkan jika dipilih)
                    feat_preview = {}
                    if show_features:
                        for k in KEY_FEATURES:
                            if k in feats:
                                feat_preview[k] = round(float(feats[k]), 4)

                    results.append({
                        "filename": f.filename,
                        "pred": pred,
                        "conf": f"{conf:.3f}",
                        "topk": [f"{lbl}:{p:.3f}" for lbl, p in topk],
                        "img_url": img_url,
                        "features": feat_preview if show_features else None
                    })

                    # simpan baris lengkap utk CSV (pakai FEATURE_COLS agar konsisten dgn model)
                    row = {"filename": f.filename, "pred": pred, "confidence": float(conf)}
                    for c in FEATURE_COLS:
                        row[c] = float(feats.get(c, 0.0))
                    rows.append(row)

                except Exception as e:
                    results.append({
                        "filename": f.filename,
                        "pred": "ERROR",
                        "conf": "-",
                        "topk": [],
                        "img_uri": None,
                        "error": str(e),
                        "features": None
                    })

            # buat CSV kalau ada baris
            if rows:
                df = pd.DataFrame(rows)
                csv_name = f"features_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
                csv_path = EXPORT_DIR / csv_name
                df.to_csv(csv_path, index=False)
                csv_relpath = f"exports/{csv_name}"  # relatif dari /static

    return render_template(
        "deteksi.html",
        results=results, error=error,
        classes=CLASSES, masked=USE_MASKED_INPUT,
        csv_relpath=csv_relpath, show_features=show_features
    )

@app.route("/api/extract", methods=["POST"])
def api_extract():
    files = request.files.getlist("files")
    if not files:
        f = request.files.get("file")
        if f: files = [f]
    if not files:
        return jsonify({"error": "unggah file dengan key 'files' atau 'file'"}), 400

    out = []
    for f in files:
        try:
            bgr = bgr_from_filestorage(f)
            feats = extract_manual_features_from_bgr(bgr)
            out.append({
                "filename": f.filename,
                "features": {k: float(feats.get(k, 0.0)) for k in FEATURE_COLS}
            })
        except Exception as e:
            out.append({"filename": f.filename, "error": str(e)})
    return jsonify({"results": out, "feature_cols": FEATURE_COLS})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    files = request.files.getlist("files")
    if not files:
        f = request.files.get("file")
        if f: files = [f]
    if not files:
        return jsonify({"error": "unggah file pakai key 'files' atau 'file'"}), 400
    out = []
    for f in files:
        try:
            bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            pred, conf, topk = predict_bgr(bgr)
            out.append({
                "filename": f.filename, "pred": pred, "confidence": conf,
                "topk": [{"label": lbl, "score": p} for lbl, p in topk]
            })
        except Exception as e:
            out.append({"filename": f.filename, "error": str(e)})
    return jsonify({"results": out, "classes": CLASSES, "masked": USE_MASKED_INPUT, "is_hybrid": IS_HYBRID})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
