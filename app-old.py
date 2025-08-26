import os, io, json, math, pickle, base64
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input as effnet_preprocess
from keras import layers, Model, Input
import keras

# ---------------------------
# Konfigurasi umum
# ---------------------------
SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

IMG_SIZE = (224, 224)
USE_MASKED_INPUT = True     # set False jika tak ingin memutihkan background
ARTIFACT_DIR = Path("artifactsv2")
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Load artefak
# ---------------------------
def load_artifacts():
    meta_path  = ARTIFACT_DIR / "meta.json"
    scaler_path = ARTIFACT_DIR / "scaler.pkl"
    le_path     = ARTIFACT_DIR / "label_encoder.pkl"
    model_path  = ARTIFACT_DIR / "best_hybrid.keras"

    if not meta_path.exists() or not scaler_path.exists() or not le_path.exists() or not model_path.exists():
        raise RuntimeError(
            "Artefak belum lengkap. Pastikan folder artifacts berisi: "
            "best_hybrid.keras, meta.json, scaler.pkl, label_encoder.pkl"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    model = keras.models.load_model(model_path)
    return model, scaler, le, meta

model, scaler, le, meta = load_artifacts()
FEATURE_COLS: List[str] = meta["feature_cols"]
CLASSES: List[str] = meta["classes"]
IMG_SIZE = tuple(meta.get("img_size", list(IMG_SIZE)))

# ---------------------------
# Utilitas fitur manual
# ---------------------------
MIN_AREA_RATIO = 0.005  # 0.5%

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
    gq = (gm/32).astype(np.uint8)  # 0..7
    glcm = graycomatrix(gq, distances=[1,2,4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    props = {}
    for prop in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
        props[prop] = float(np.mean(graycoprops(glcm, prop)))
    lbp = local_binary_pattern(gm, P=8, R=1, method="uniform")
    lbp_masked = lbp[mask>0]
    n_bins = 8 + 2
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

# ---------------------------
# Preprocess gambar untuk CNN (EfficientNet)
# ---------------------------
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

# ---------------------------
# Prediksi
# ---------------------------
def predict_bgr(bgr: np.ndarray) -> Tuple[str, float, List[Tuple[str,float]]]:
    # fitur manual
    feats = extract_manual_features_from_bgr(bgr)
    vec = np.array([[feats.get(c, 0.0) for c in FEATURE_COLS]], dtype=np.float32)
    vec = scaler.transform(vec)

    # gambar
    im = preprocess_for_model(bgr)
    im = np.expand_dims(im, axis=0)

    # infer
    probs = model.predict([im, vec], verbose=0)[0]
    top_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([top_idx])[0]
    top1 = float(probs[top_idx])
    # top-3
    order = np.argsort(probs)[::-1][:3]
    topk = [(le.classes_[i], float(probs[i])) for i in order]
    return pred_label, top1, topk

def bgr_from_filestorage(file) -> np.ndarray:
    # baca bytes → np.uint8 → decode OpenCV
    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Gagal membaca gambar (format tidak didukung).")
    return bgr

def to_data_uri(bgr: np.ndarray) -> str:
    # tampilkan Gambar prediksi (tanpa mengubah dimensi)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def annotated_copy(bgr: np.ndarray, label: str, conf: float) -> np.ndarray:
    out = bgr.copy()
    txt = f"{label} ({conf:.2f})"
    cv2.putText(out, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
    return out

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            error = "Pilih minimal satu gambar."
        else:
            for f in files:
                try:
                    bgr = bgr_from_filestorage(f)
                    pred, conf, topk = predict_bgr(bgr)
                    # buat versi anotasi utk display
                    anno = annotated_copy(bgr, pred, conf)
                    img_uri = to_data_uri(anno)
                    results.append({
                        "filename": f.filename,
                        "pred": pred,
                        "conf": f"{conf:.3f}",
                        "topk": [f"{lbl}:{p:.3f}" for lbl, p in topk],
                        "img_uri": img_uri
                    })
                except Exception as e:
                    results.append({
                        "filename": f.filename,
                        "pred": "ERROR",
                        "conf": "-",
                        "topk": [],
                        "img_uri": None,
                        "error": str(e)
                    })
    return render_template("index.html", results=results, error=error, classes=CLASSES, masked=USE_MASKED_INPUT)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # multipart (files=...) atau satu file dengan key 'file'
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
            pred, conf, topk = predict_bgr(bgr)
            out.append({
                "filename": f.filename,
                "pred": pred,
                "confidence": conf,
                "topk": [{"label": lbl, "score": p} for lbl, p in topk]
            })
        except Exception as e:
            out.append({"filename": f.filename, "error": str(e)})
    return jsonify({"results": out, "classes": CLASSES, "masked": USE_MASKED_INPUT})

if __name__ == "__main__":
    # jalankan: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
