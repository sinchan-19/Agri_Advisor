"""
Krishi Mitra (ಕೃಷಿ ಮಿತ್ರ / कृषि मित्र) - Production AI Server
AI-Powered Crop Doctor & Precision Agronomy Advisory Engine
Features:
- Dual-Stage Maize Verification Gate (MobileNetV2)
- Multi-Class Maize Disease Detection (Deep CNN)
- Grad-CAM Spatial Attention Heatmaps (conv2d_3 layer)
- Native Multilingual Advisory in English, Kannada (ಕನ್ನಡ), and Hindi (हिन्दी)
- Real-world 16L Knapsack Sprayer & Land Acreage Dosage Calculations
- Voice-enabled AI Agronomist Chatbot
"""

import base64
import io
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import matplotlib as mpl
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================================================
# FLASK CONFIGURATION & BASE PATHS
# =========================================================

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

def _resolve_model_path(*filenames):
    """Finds the first existing model file in ai_model directory."""
    for name in filenames:
        candidate = BASE_DIR / "ai_model" / name
        if candidate.exists():
            return candidate
    return BASE_DIR / "ai_model" / filenames[0]

def _is_lfs_pointer(path):
    """Checks if a file is a Git LFS pointer text file instead of binary."""
    try:
        if path.is_file() and path.stat().st_size < 1000:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(100)
                return "git-lfs" in content or "oid sha256" in content
    except Exception:
        pass
    return False

GATE_MODEL_PATH = _resolve_model_path("maize_gate_v2.keras", "maize_gate.keras")
DISEASE_MODEL_PATH = _resolve_model_path("maize_disease_model_best.keras", "maize_disease_model.keras")
CLASS_FILE_PATH = BASE_DIR / "ai_model" / "classes.txt"

MAIZE_GATE_THRESHOLD = 0.50
IMAGE_SIZE = (224, 224)

# =========================================================
# LOAD NEURAL NETWORK MODELS
# =========================================================

print("[Krishi Mitra AI] Initializing Neural Models...")

# 1. Maize Gate Model
maize_gate = None
if not GATE_MODEL_PATH.exists():
    print(f"[-] Warning: Gate model not found at {GATE_MODEL_PATH}")
elif _is_lfs_pointer(GATE_MODEL_PATH):
    print(f"[-] Error: Gate model is a Git LFS pointer. Run 'git lfs pull'.")
else:
    try:
        maize_gate = tf.keras.models.load_model(GATE_MODEL_PATH)
        print(f"[+] Loaded Maize Gate Model: {GATE_MODEL_PATH.name}")
    except Exception as e:
        print(f"[-] Failed to load Maize Gate: {e}")

# 2. Disease Classification Model
disease_model = None
if not DISEASE_MODEL_PATH.exists():
    print(f"[-] Warning: Disease model not found at {DISEASE_MODEL_PATH}")
elif _is_lfs_pointer(DISEASE_MODEL_PATH):
    print(f"[-] Error: Disease model is a Git LFS pointer. Run 'git lfs pull'.")
else:
    try:
        disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
        print(f"[+] Loaded Disease Model: {DISEASE_MODEL_PATH.name}")
    except Exception as e:
        print(f"[-] Failed to load Disease Model: {e}")

# 3. Disease Classes
classes = []
if CLASS_FILE_PATH.exists():
    with open(CLASS_FILE_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
if not classes:
    classes = ["common_rust", "gray_leaf_spot", "healthy", "northern_leaf_blight"]

print(f"[+] Active Disease Classes: {classes}")

# =========================================================
# COMPREHENSIVE MULTILINGUAL AGRONOMIC DATABASE
# (English, Kannada, Hindi + Real-World Knapsack Pump Calculations)
# =========================================================

DISEASE_DATABASE = {
    "common_rust": {
        "en": {
            "name": "Common Rust",
            "pathogen": "Puccinia sorghi (Fungus)",
            "cause": "Caused by the fungus Puccinia sorghi. Airborne urediniospores spread rapidly under moderate temperatures (16°C–25°C), high relative humidity (>95%), and prolonged leaf dampness (dew or light rain for 6+ continuous hours).",
            "symptoms": "Small, oval, cinnamon-brown to reddish-brown powdery pustules on both upper and lower leaf surfaces. Pustules rupture the epidermis, eventually turning brownish-black.",
            "prevention": "1. Sow certified rust-resistant maize hybrid seeds.\n2. Maintain recommended row spacing (60 cm x 20-25 cm) for optimal canopy aeration.\n3. Avoid late-evening overhead sprinkler irrigation.\n4. Rotate fields with non-cereal crops (soybean, groundnut, pulses) for 1–2 seasons.\n5. Deep plow crop residue after harvest.",
            "treatment": "Apply protective fungicide at first sign of pustules on lower leaves. For severe outbreaks, use systemic triazole or strobilurin fungicides.",
            "dosage": "• Mancozeb 75% WP: 2.0 to 2.5 g/L water (500 g/acre in 200 L water).\n• Propiconazole 25% EC (Tilt): 1.0 mL/L water (200 mL/acre in 200 L water).\n• Azoxystrobin + Difenoconazole: 1.0 mL/L water (200 mL/acre).\n• Organic Option: Neem oil (1500 ppm) @ 3–5 mL/L OR Pseudomonas fluorescens @ 5 g/L.",
            "timing": "Spray early morning (06:30 AM – 09:30 AM). Repeat after 10–14 days if humid weather persists.",
            "yield": "Untreated rust can cause 15%–30% grain loss. Timely treatment protects ear-leaf filling capacity.",
            "solution": "Spray Mancozeb 75% WP @ 40g per 16L knapsack pump (500g/acre) or Tilt @ 16mL per 16L pump."
        },
        "kn": {
            "name": "ಕಾಮನ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ)",
            "pathogen": "ಪುಕ್ಸೀನಿಯಾ ಸೊರ್ಘಿ (ಶಿಲೀಂಧ್ರ / Puccinia sorghi)",
            "cause": "ಇದು ಪುಕ್ಸೀನಿಯಾ ಸೊರ್ಘಿ ಎಂಬ ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ. ತಂಪಾದ ವಾತಾವರಣ (16°C–25°C), ಅಧಿಕ ತೇವಾಂಶ (>95%) ಮತ್ತು ಎಲೆಗಳ ಮೇಲೆ 6 ಗಂಟೆಗಿಂತ ಹೆಚ್ಚು ಕಾಲ ನೀರು ನಿಲ್ಲುವುದರಿಂದ ಗಾಳಿಯ ಮೂಲಕ ಬೀಜಕಗಳು ಹರಡುತ್ತವೆ.",
            "symptoms": "ಎಲೆಯ ಮೇಲ್ಭಾಗ ಮತ್ತು ಕೆಳಭಾಗದಲ್ಲಿ ಸಣ್ಣ, ಕಂದು ಬಣ್ಣದ ಹುಡಿಯುಳ್ಳ ಗುಳ್ಳೆಗಳು (pustules) ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತವೆ. ರೋಗ ಉಲ್ಬಣಿಸಿದಂತೆ ಇವು ಕಪ್ಪು-ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಎಲೆ ಒಣಗುತ್ತದೆ.",
            "prevention": "1. ರೋಗ ನಿರೋಧಕ ಹೈಬ್ರಿಡ್ ಬೀಜಗಳನ್ನು ಬಿತ್ತನೆ ಮಾಡಿ.\n2. ಸಾಲಿನಿಂದ ಸಾಲಿಗೆ 60 ಸೆಂ.ಮೀ, ಗಿಡದಿಂದ ಗಿಡಕ್ಕೆ 20-25 ಸೆಂ.ಮೀ ಅಂತರ ಕಾಪಾಡಿ.\n3. ಸಂಜೆ ವೇಳೆ ನೀರು ಹಾಯಿಸಬೇಡಿ; ಹೊಲದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಕಾಲುವೆ ನಿರ್ಮಿಸಿ.\n4. ದ್ವಿದಳ ಧಾನ್ಯಗಳೊಂದಿಗೆ (ಸೋಯಾಬೀನ್/ಕಡಲೆಕಾಯಿ) ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ.\n5. ಕೊಯ್ಲಿನ ನಂತರ ಹಳೆಯ ಕಸವನ್ನು ಆಳವಾಗಿ ಉಳುಮೆ ಮಾಡಿ.",
            "treatment": "ಕೆಳಗಿನ ಎಲೆಗಳಲ್ಲಿ ಕಂದು ಗುಳ್ಳೆಗಳು ಕಂಡ ತಕ್ಷಣ ರಕ್ಷಣಾತ್ಮಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ.",
            "dosage": "• ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP: 2.5 ಗ್ರಾಂ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 40 ಗ್ರಾಂ / ಎಕರೆಗೆ 500 ಗ್ರಾಂ).\n• ಪ್ರಾಪಿಕೊನಜೋಲ್ 25% EC (ಟಿಲ್ಟ್): 1.0 ಮಿ.ಲೀ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 16 ಮಿ.ಲೀ / ಎಕರೆಗೆ 200 ಮಿ.ಲೀ).\n• ಅಜೋಕ್ಸಿಸ್ಟ್ರೋಬಿನ್ + ಡೈಫೆನೊಕೊನಜೋಲ್: 1.0 ಮಿ.ಲೀ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ.\n• ಜೈವಿಕ ಪರಿಹಾರ: ಬೇವಿನ ಎಣ್ಣೆ (1500 ppm) @ 3-5 ಮಿ.ಲೀ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ.",
            "timing": "ಬೆಳಿಗ್ಗೆ 06:30 ರಿಂದ 09:30 ರೊಳಗೆ ಸಿಂಪಡಿಸಿ. ಮೋಡ ಕವಿದ ವಾತಾವರಣ ಮುಂದುವರಿದರೆ 10-14 ದಿನಗಳ ನಂತರ ಪುನರಾವರ್ತಿಸಿ.",
            "yield": "ಸಕಾಲದಲ್ಲಿ ಚಿಕಿತ್ಸೆ ನೀಡದಿದ್ದರೆ ಶೇ. 15 ರಿಂದ 30 ರಷ್ಟು ಇಳುವರಿ ನಷ್ಟವಾಗಬಹುದು. ತೆನೆ ಕಟ್ಟುವ ಎಲೆಗಳನ್ನು ರಕ್ಷಿಸುವುದು ಮುಖ್ಯ.",
            "solution": "16 ಲೀಟರ್ ಸ್ಪ್ರೇ ಪಂಪ್‌ಗೆ 40 ಗ್ರಾಂ ಮ್ಯಾಂಕೋಜೆಬ್ ಅಥವಾ 16 ಮಿ.ಲೀ ಟಿಲ್ಟ್ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        },
        "hi": {
            "name": "कॉमन रस्ट (रतुआ रोग)",
            "pathogen": "पुकिनिया सोर्गी (कवक / Puccinia sorghi)",
            "cause": "यह पुकिनिया सोर्गी नामक कवक से होता है। 16°C से 25°C का तापमान, 95% से अधिक सापेक्षिक आर्द्रता और पत्तियों पर 6 घंटे से अधिक ओस रहने पर यह हवा द्वारा तेजी से फैलता है।",
            "symptoms": "पत्तियों के दोनों तरफ छोटे, अंडाकार, दालचीनी जैसे भूरे रंग के चूर्णयुक्त फफोले बनते हैं, जो बाद में गहरे काले-भूरे होकर पत्ती को सुखा देते हैं।",
            "prevention": "1. प्रमाणित रोगरोधी संकर बीजों की बुवाई करें।\n2. कतार से कतार 60 सेमी और पौधे से पौधे 20 सेमी की दूरी रखें।\n3. शाम के समय फव्वारा सिंचाई से बचें ताकि पत्तियां रातभर गीली न रहें।\n4. दलहनी फसलों के साथ 1-2 वर्ष का फसल चक्र अपनाएं।\n5. कटाई के बाद खेत की गहरी जुताई करें।",
            "treatment": "निचली पत्तियों पर लक्षण दिखते ही कवकनाशी का समय पर छिड़काव करें।",
            "dosage": "• मैंकोजेब 75% WP: 2.5 ग्राम प्रति लीटर पानी (16 लीटर पंप में 40 ग्राम / प्रति एकड़ 500 ग्राम)।\n• प्रोपिकोनाज़ोल 25% EC (टिल्ट): 1.0 मिली प्रति लीटर पानी (16 लीटर पंप में 16 मिली / प्रति एकड़ 200 मिली)।\n• एज़ोक्सीस्ट्रोबिन + डिफेनोकोनाज़ोल: 1.0 मिली प्रति लीटर पानी।\n• जैविक उपाय: नीम का तेल (1500 ppm) @ 3-5 मिली प्रति लीटर पानी।",
            "timing": "सुबह 6:30 से 9:30 के बीच शांत मौसम में छिड़कें। आवश्यकता पड़ने पर 10-14 दिनों बाद दोहराएं।",
            "yield": "उपचार न करने पर 15% से 30% तक पैदावार घट सकती है। भुट्टे की पोषण वाली पत्तियों की सुरक्षा अत्यंत आवश्यक है।",
            "solution": "16 लीटर नेपसैक स्प्रेयर में 40 ग्राम मैंकोजेब या 16 मिली टिल्ट मिलाकर अच्छी तरह छिड़काव करें।"
        },
        "calc": {
            "chemical_name": "Mancozeb 75% WP (or Tilt 25% EC)",
            "dose_per_liter_g": 2.5,
            "dose_per_16L_pump_g": 40.0,
            "water_per_acre_L": 200,
            "pumps_per_acre_16L": 12.5,
            "total_pack_needed_g": 500,
            "approx_cost_inr": 180
        }
    },
    "northern_leaf_blight": {
        "en": {
            "name": "Northern Leaf Blight",
            "pathogen": "Exserohilum turcicum (Fungus)",
            "cause": "Caused by Exserohilum turcicum. Pathogen survives in infected corn stover on the soil surface and is dispersed by wind and rain splash during moderate temperatures (18°C–27°C) with 6–18 hours of continuous dew or rainfall.",
            "symptoms": "Long, narrow, elliptical or cigar-shaped grayish-green to tan lesions (2.5 to 15 cm long). Under humid conditions, dark olive-green velvety fungal sporulation forms in lesion centers, coalescing to blight entire leaves.",
            "prevention": "1. Sow certified hybrids with genetic Ht-resistance.\n2. Practice 1–2 year crop rotation with non-cereal crops.\n3. Deep plow residues to accelerate decomposition.\n4. Avoid excessive nitrogen; ensure adequate potassium and zinc.",
            "treatment": "Apply systemic triazole fungicides when lesions first appear on lower canopy before tasseling.",
            "dosage": "• Propiconazole 25% EC (Tilt): 1.0 mL/L water (200 mL/acre in 200 L water).\n• Tebuconazole 25.9% EC: 1.0 to 1.25 mL/L water (200–250 mL/acre).\n• Mancozeb 75% WP: 2.5 g/L water (500 g/acre).\n• Organic Option: Pseudomonas fluorescens @ 5 g/L OR Bacillus subtilis @ 5 g/L.",
            "timing": "Spray immediately when cigar-shaped lesions appear on middle leaves before silking. Repeat in 12–15 days if humid.",
            "yield": "Can cause 30%–50% yield reduction if widespread prior to pollination. Protects ear leaf fill.",
            "solution": "Spray Propiconazole 25% EC @ 16 mL per 16L pump (200 mL/acre) or Mancozeb @ 40g per 16L pump."
        },
        "kn": {
            "name": "ಉತ್ತರ ಎಲೆ ಅಂಗಮಾರಿ ರೋಗ (NLB)",
            "pathogen": "ಎಕ್ಸರೋಹಿಲಮ್ ಟರ್ಸಿಕಮ್ (ಶಿಲೀಂಧ್ರ / Exserohilum turcicum)",
            "cause": "ಇದು ಎಕ್ಸರೋಹಿಲಮ್ ಟರ್ಸಿಕಮ್ ಎಂಬ ಶಿಲೀಂಧ್ರದಿಂದ ಬರುತ್ತದೆ. ಮಣ್ಣಿನಲ್ಲಿರುವ ಹಳೆಯ ಬೆಳೆ ಕಸದಲ್ಲಿ ಉಳಿಯುವ ಬೀಜಕಗಳು ಮಳೆ ಹನಿ ಮತ್ತು ಗಾಳಿಯಿಂದ ಹರಡುತ್ತವೆ (18°C–27°C ತಾಪಮಾನ ಮತ್ತು ತೇವಾಂಶ ಇದಕ್ಕೆ ಪೂರಕ).",
            "symptoms": "ಎಲೆಗಳ ಮೇಲೆ ಉದ್ದನೆಯ, ಸಿಗಾರ್ ಅಥವಾ ದೋಣಿ ಆಕಾರದ ಬೂದು-ಹಸಿರು ಅಥವಾ ಕಂದು ಬಣ್ಣದ ಕಲೆಗಳು (2.5 ರಿಂದ 15 ಸೆಂ.ಮೀ ಉದ್ದ) ಉಂಟಾಗುತ್ತವೆ. ತೇವಾಂಶವಿದ್ದಾಗ ಕಲೆಗಳ ಮಧ್ಯೆ ಕಪ್ಪು ಬೂಜು ಬೆಳೆಯುತ್ತದೆ.",
            "prevention": "1. ರೋಗ ನಿರೋಧಕ Ht-ತಳಿಗಳನ್ನು ಬೆಳೆಯಿರಿ.\n2. ಕನಿಷ್ಠ 1-2 ವರ್ಷ ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ.\n3. ಕೊಯ್ಲಿನ ನಂತರ ಹಳೆಯ ಬೆಳೆ ಕಸವನ್ನು ಆಳವಾಗಿ ಉಳುಮೆ ಮಾಡಿ ಮಣ್ಣಿನಲ್ಲಿ ಸೇರಿಸಿ.\n4. ಅತಿಯಾದ ಯೂರಿಯಾ ತಪ್ಪಿಸಿ; ಸಮತೋಲಿತ ಪೊಟ್ಯಾಶ್ ಮತ್ತು ಜಿಂಕ್ ನೀಡಿ.",
            "treatment": "ತೆನೆ ಮೂಡುವ ಮುನ್ನ ಎಲೆಗಳಲ್ಲಿ ಉದ್ದನೆಯ ಕಲೆಗಳು ಕಂಡ ತಕ್ಷಣ ಪ್ರಾಪಿಕೊನಜೋಲ್ ಸಿಂಪಡಿಸಿ.",
            "dosage": "• ಪ್ರಾಪಿಕೊನಜೋಲ್ 25% EC (ಟಿಲ್ಟ್): 1.0 ಮಿ.ಲೀ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 16 ಮಿ.ಲೀ / ಎಕರೆಗೆ 200 ಮಿ.ಲೀ).\n• ಟೆಬುಕೊನಜೋಲ್ 25.9% EC: 1.0 ರಿಂದ 1.25 ಮಿ.ಲೀ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ.\n• ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP: 2.5 ಗ್ರಾಂ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 40 ಗ್ರಾಂ).\n• ಜೈವಿಕ ಪರಿಹಾರ: ಸೂಡೊಮೊನಾಸ್ ಫ್ಲೋರೊಸೆನ್ಸ್ @ 5 ಗ್ರಾಂ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ.",
            "timing": "ತೆನೆ ಬರುವ ಮುನ್ನ ಸಿಗಾರ್ ಆಕಾರದ ಕಲೆಗಳು ಕಂಡ ತಕ್ಷಣ ಬೆಳಿಗ್ಗೆ ಸಿಂಪಡಿಸಿ. 12-15 ದಿನಗಳ ನಂತರ ಪುನರಾವರ್ತಿಸಿ.",
            "yield": "ಸಕಾಲದಲ್ಲಿ ನಿಯಂತ್ರಿಸದಿದ್ದರೆ ಶೇ. 30 ರಿಂದ 50 ರಷ್ಟು ಇಳುವರಿ ಕುಂಠಿತವಾಗಬಹುದು.",
            "solution": "16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 16 ಮಿ.ಲೀ ಪ್ರಾಪಿಕೊನಜೋಲ್ (ಟಿಲ್ಟ್) ಅಥವಾ 40 ಗ್ರಾಂ ಮ್ಯಾಂಕೋಜೆಬ್ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        },
        "hi": {
            "name": "उत्तरी पत्ती झुलसा रोग (NLB)",
            "pathogen": "एक्सरोहिलम टर्सिकम (कवक / Exserohilum turcicum)",
            "cause": "यह एक्सरोहिलम टर्सिकम कवक के कारण होता है। यह जमीन पर पड़े पुराने फसल अवशेषों में जीवित रहता है और 18°C से 27°C तापमान और 6-18 घंटे की लगातार नमी में बारिश की बौछारों से फैलता है।",
            "symptoms": "पत्तियों पर सिगार के आकार के लंबे (2.5 से 15 सेमी), संकरे, धूसर-भूरे रंग के धब्बे बनते हैं, जो आपस में मिलकर पूरी पत्ती को झुलसा देते हैं।",
            "prevention": "1. Ht-प्रतिरोधी संकर बीजों का चयन करें।\n2. गैर-घास कुल की फसलों के साथ 1-2 वर्ष का फसल चक्र अपनाएं।\n3. फसल कटाई के बाद गहरी जुताई करें।\n4. संतुलित खाद दें; पोटाश और जिंक की कमी न होने दें।",
            "treatment": "भुट्टा निकलने से पहले पत्तियों पर धब्बे दिखते ही व्यवस्थित कवकनाशी का छिड़काव करें।",
            "dosage": "• प्रोपिकोनाज़ोल 25% EC (टिल्ट): 1.0 मिली प्रति लीटर पानी (16 लीटर पंप में 16 मिली / प्रति एकड़ 200 मिली)।\n• टेबुकोनाज़ोल 25.9% EC: 1.0 से 1.25 मिली प्रति लीटर पानी।\n• मैंकोजेब 75% WP: 2.5 ग्राम प्रति लीटर पानी (16 लीटर पंप में 40 ग्राम)।\n• जैविक विकल्प: स्यूडोमोनास फ्लोरोसेंस @ 5 ग्राम प्रति लीटर पानी।",
            "timing": "पत्तियों पर लंबे धब्बे दिखते ही सुबह के समय छिड़कें। 12-15 दिनों बाद आवश्यकतानुसार दोहराएं।",
            "yield": "फूल आने से पहले गंभीर प्रकोप होने पर 30% से 50% तक उपज का भारी नुकसान हो सकता है।",
            "solution": "16 लीटर स्प्रे पंप में 16 मिली प्रोपिकोनाज़ोल (टिल्ट) या 40 ग्राम मैंकोजेब मिलाकर छिड़काव करें।"
        },
        "calc": {
            "chemical_name": "Propiconazole 25% EC (Tilt)",
            "dose_per_liter_g": 1.0,
            "dose_per_16L_pump_g": 16.0,
            "water_per_acre_L": 200,
            "pumps_per_acre_16L": 12.5,
            "total_pack_needed_g": 200,
            "approx_cost_inr": 280
        }
    },
    "gray_leaf_spot": {
        "en": {
            "name": "Gray Leaf Spot",
            "pathogen": "Cercospora zeae-maydis (Fungus)",
            "cause": "Caused by Cercospora zeae-maydis, surviving on surface crop residues. Favored by warm temperatures (25°C–32°C), high relative humidity (>90%), overcast skies, and continuous monoculture.",
            "symptoms": "Distinct rectangular, tan-to-gray lesions strictly bounded between leaf veins with parallel edges. Lesions migrate upward to the ear-leaf and upper canopy.",
            "prevention": "1. Plant resistant maize hybrids.\n2. Rotate with non-host crops (soybean, sunflower, cotton) for 1–2 seasons.\n3. Incorporate corn stubble via tillage to accelerate decomposition.\n4. Avoid excessive nitrogen without potassium.",
            "treatment": "Apply strobilurin or triazole fungicides around tasseling/silking stage.",
            "dosage": "• Pyraclostrobin 20% WG: 1.0 g/L water (200 g/acre in 200 L water).\n• Carbendazim + Mancozeb (Saaf): 2.0 g/L water (400 g/acre in 200 L water).\n• Azoxystrobin 23% SC: 1.0 mL/L water (200 mL/acre).\n• Organic Option: Trichoderma harzianum @ 5 g/L water.",
            "timing": "Spray between V12 and silking if rectangular lesions appear on or just below ear leaf.",
            "yield": "Causes premature leaf death, stalk lodging, and 40%–50% yield reduction if untreated.",
            "solution": "Spray Pyraclostrobin @ 16g per 16L pump (200g/acre) or Saaf @ 32g per 16L pump."
        },
        "kn": {
            "name": "ಗ್ರೇ ಲೀಫ್ ಸ್ಪಾಟ್ (ಬೂದು ಎಲೆ ಚುಕ್ಕೆ)",
            "pathogen": "ಸರ್ಕೋಸ್ಪೋರಾ ಜಿಯಾ-ಮೇಡಿಸ್ (ಶಿಲೀಂಧ್ರ / Cercospora zeae-maydis)",
            "cause": "ಇದು ಸರ್ಕೋಸ್ಪೋರಾ ಜಿಯಾ-ಮೇಡಿಸ್ ಶಿಲೀಂಧ್ರದಿಂದ ಬರುತ್ತದೆ. ಬೆಚ್ಚನೆಯ ಹವಾಮಾನ (25°C–32°C), ನಿರಂತರ ಮೋಡ ಮತ್ತು ಸತತವಾಗಿ ಮೆಕ್ಕೆಜೋಳ ಬೆಳೆಯುವುದರಿಂದ ಇದು ಉಂಟಾಗುತ್ತದೆ.",
            "symptoms": "ಎಲೆಯ ನರಗಳ ನಡುವೆ ಕಟ್ಟುನಿಟ್ಟಾಗಿ ಸೀಮಿತವಾದ ಚೌಕಾಕಾರದ (rectangular), ಬೂದು-ಕಂದು ಬಣ್ಣದ ಕಲೆಗಳು ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತವೆ. ಕೆಳಗಿನಿಂದ ಮೇಲಕ್ಕೆ ಹರಡಿ ಎಲೆ ಸುಟ್ಟುಹೋಗುತ್ತದೆ.",
            "prevention": "1. ರೋಗ ಸಹಿಷ್ಣು ಹೈಬ್ರಿಡ್ ಬೀಜ ಬೆಳೆಯಿರಿ.\n2. ಕನಿಷ್ಠ 1 ವರ್ಷ ದ್ವಿದಳ ಧಾನ್ಯಗಳೊಂದಿಗೆ ಬೆಳೆ ಬದಲಿಸಿ.\n3. ಕಸವನ್ನು ಮಣ್ಣಿನಲ್ಲಿ ಆಳವಾಗಿ ಉಳುಮೆ ಮಾಡಿ.\n4. ಸಮತೋಲಿತ ರಸಗೊಬ್ಬರ ನೀಡಿ.",
            "treatment": "ತೆನೆ ಎಲೆಯ ಕೆಳಗೆ ಚೌಕಾಕಾರದ ಕಲೆಗಳು ಕಂಡಾಗ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ.",
            "dosage": "• ಪೈರಾಕ್ಲೋಸ್ಟ್ರೋಬಿನ್ 20% WG: 1.0 ಗ್ರಾಂ/ಲೀ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 16 ಗ್ರಾಂ / ಎಕರೆಗೆ 200 ಗ್ರಾಂ).\n• ಸಾಫ್ (ಕಾರ್ಬೆಂಡಾಜಿಮ್ + ಮ್ಯಾಂಕೋಜೆಬ್): 2.0 ಗ್ರಾಂ/ಲೀ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 32 ಗ್ರಾಂ / ಎಕರೆಗೆ 400 ಗ್ರಾಂ).\n• ಅಜೋಕ್ಸಿಸ್ಟ್ರೋಬಿನ್ 23% SC: 1.0 ಮಿ.ಲೀ/ಲೀ.\n• ಜೈವಿಕ ಪರಿಹಾರ: ಟ್ರೈಕೋಡರ್ಮಾ ಹಾರ್ಜಿಯಾನಮ್ @ 5 ಗ್ರಾಂ ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ.",
            "timing": "ತೆನೆ ಕಟ್ಟುವ ಹಂತದಲ್ಲಿ ಚೌಕಾಕಾರದ ಕಲೆಗಳು ಕಂಡ ತಕ್ಷಣ ಬೆಳಿಗ್ಗೆ ಸಿಂಪಡಿಸಿ.",
            "yield": "ನಿರ್ಲಕ್ಷಿಸಿದರೆ ಶೇ. 40 ರಿಂದ 50 ರಷ್ಟು ಇಳುವರಿ ನಷ್ಟ ಮತ್ತು ಗಿಡಗಳು ಉರುಳಿ ಬೀಳುವ (lodging) ಅಪಾಯವಿದೆ.",
            "solution": "16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 16 ಗ್ರಾಂ ಪೈರಾಕ್ಲೋಸ್ಟ್ರೋಬಿನ್ ಅಥವಾ 32 ಗ್ರಾಂ ಸಾಫ್ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        },
        "hi": {
            "name": "ग्रे लीफ स्पॉट (धूसर पत्ती धब्बा)",
            "pathogen": "सर्कोस्पोरा ज़िया-मेडिस (कवक / Cercospora zeae-maydis)",
            "cause": "यह सर्कोस्पोरा ज़िया-मेडिस कवक से होता है। 25°C से 32°C का गर्म तापमान, 90% से अधिक नमी और लगातार मक्के की खेती इसके मुख्य कारण हैं।",
            "symptoms": "पत्तियों की नसों के बीच सीमित आयताकार (rectangular) भूरे-धूसर धब्बे बनते हैं जो नीचे से ऊपर की ओर फैलकर पूरी पत्ती को सुखा देते हैं।",
            "prevention": "1. प्रतिरोधी संकर किस्मों का चयन करें।\n2. सोयाबीन या सूरजमुखी के साथ फसल चक्र अपनाएं।\n3. फसल अवशेषों को मिट्टी में दबाने के लिए गहरी जुताई करें।\n4. संतुलित खाद दें।",
            "treatment": "भुट्टे की पत्ती के नीचे आयताकार धब्बे दिखने पर कवकनाशी का छिड़काव करें।",
            "dosage": "• पायराक्लोस्ट्रोबिन 20% WG: 1.0 ग्राम प्रति लीटर पानी (16 लीटर पंप में 16 ग्राम / प्रति एकड़ 200 ग्राम)।\n• साफ (कार्बेन्डाजिम + मैंकोजेब): 2.0 ग्राम प्रति लीटर पानी (16 लीटर पंप में 32 ग्राम / प्रति एकड़ 400 ग्राम)।\n• एज़ोक्सीस्ट्रोबिन 23% SC: 1.0 मिली प्रति लीटर पानी।\n• जैविक उपाय: ट्राइकोडर्मा हार्ज़ियानम @ 5 ग्राम प्रति लीटर पानी।",
            "timing": "भुट्टा बनने से पहले आयताकार धब्बे दिखते ही छिड़काव करें।",
            "yield": "गंभीर अवस्था में 40% से 50% तक उपज का भारी नुकसान और तना गिरने की समस्या हो सकती है।",
            "solution": "16 लीटर नेपसैक स्प्रेयर में 16 ग्राम पायराक्लोस्ट्रोबिन या 32 ग्राम साफ मिलाकर छिड़कें।"
        },
        "calc": {
            "chemical_name": "Pyraclostrobin 20% WG (or Saaf)",
            "dose_per_liter_g": 1.0,
            "dose_per_16L_pump_g": 16.0,
            "water_per_acre_L": 200,
            "pumps_per_acre_16L": 12.5,
            "total_pack_needed_g": 200,
            "approx_cost_inr": 340
        }
    },
    "healthy": {
        "en": {
            "name": "Healthy Maize",
            "pathogen": "None detected",
            "cause": "No pathogenic infection or visible foliar lesions detected. Leaf exhibits normal cellular structure and uniform green chlorophyll pigmentation.",
            "symptoms": "Smooth, uniform green leaf surface without pustules, water-soaked spots, or necrotic margins.",
            "prevention": "1. Maintain balanced fertilization: NPK 120:60:40 kg/ha with split nitrogen top-dressing.\n2. Keep fields weed-free during first 35 days.\n3. Ensure critical irrigation at flowering (silking) and grain-filling.\n4. Scout weekly for Fall Armyworm.",
            "treatment": "No chemical fungicide intervention required.",
            "dosage": "• Preventive Bio-Care: Spray Pseudomonas fluorescens @ 2 g/L or Neem Oil (1500 ppm) @ 3 mL/L water to maintain foliar health.",
            "timing": "Standard seasonal agronomic routine.",
            "yield": "Healthy canopy maximizes photosynthetic capacity and supports full target yields of 6–8 tonnes/ha.",
            "solution": "Crop canopy is healthy! Continue proper irrigation, split fertilizer application, and monitoring."
        },
        "kn": {
            "name": "ಆರೋಗ್ಯಕರ ಮೆಕ್ಕೆಜೋಳ (Healthy Maize)",
            "pathogen": "ಯಾವುದೇ ರೋಗಾಣು ಪತ್ತೆಯಾಗಿಲ್ಲ",
            "cause": "ಯಾವುದೇ ಶಿಲೀಂಧ್ರ ಅಥವಾ ರೋಗದ ಲಕ್ಷಣಗಳಿಲ್ಲ. ಎಲೆಯು ನೈಸರ್ಗಿಕ ಹಸಿರು ಕ್ಲೋರೊಫಿಲ್ ಮತ್ತು ಉತ್ತಮ ಆರೋಗ್ಯವನ್ನು ಹೊಂದಿದೆ.",
            "symptoms": "ಸ್ಪಷ್ಟ, ನುಣುಪಾದ ಹಸಿರು ಎಲೆ. ಯಾವುದೇ ಕಲೆಗಳು ಅಥವಾ ಗುಳ್ಳೆಗಳಿಲ್ಲ.",
            "prevention": "1. ಸಮತೋಲಿತ NPK (120:60:40 kg/ha) ರಸಗೊಬ್ಬರ ನೀಡಿ.\n2. ಮೊದಲ 35 ದಿನಗಳಲ್ಲಿ ಕಳೆ ನಿಯಂತ್ರಿಸಿ.\n3. ಹೂವು ಬಿಡುವ ಮತ್ತು ಕಾಳು ತುಂಬುವ ಹಂತಗಳಲ್ಲಿ ಸಮರ್ಪಕ ನೀರಾವರಿ ನೀಡಿ.\n4. ಸೈನಿಕ ಹುಳು ಬಾಧೆಗಾಗಿ ವಾರಕ್ಕೊಮ್ಮೆ ಹೊಲ ಪರಿಶೀಲಿಸಿ.",
            "treatment": "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಸಿಂಪರಣೆ ಅಗತ್ಯವಿಲ್ಲ.",
            "dosage": "• ರಕ್ಷಣಾತ್ಮಕ ಜೈವಿಕ ಆರೈಕೆ: ಸೂಡೊಮೊನಾಸ್ @ 2 ಗ್ರಾಂ/ಲೀ ಅಥವಾ ಬೇವಿನ ಎಣ್ಣೆ (1500 ppm) @ 3 ಮಿ.ಲೀ/ಲೀ ಸಿಂಪಡಿಸಿ ರೋಗ ನಿರೋಧಕತೆ ಹೆಚ್ಚಿಸಿ.",
            "timing": "ವಾಡಿಕೆಯ ಕೃಷಿ ವೇಳಾಪಟ್ಟಿ ಅನುಸರಿಸಿ.",
            "yield": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದ್ದು ಗರಿಷ್ಠ ಇಳುವರಿ (ಹೆಕ್ಟೇರಿಗೆ 6-8 ಟನ್) ಪಡೆಯಲು ಪೂರಕವಾಗಿದೆ.",
            "solution": "ಬೆಳೆ ಅತ್ಯುತ್ತಮ ಸ್ಥಿತಿಯಲ್ಲಿದೆ! ನೀರಿನ ನಿರ್ವಹಣೆ ಮತ್ತು ಯೂರಿಯಾ ಮೇಲುಗೊಬ್ಬರ ಮುಂದುವರಿಸಿ."
        },
        "hi": {
            "name": "स्वस्थ मक्का (Healthy Maize)",
            "pathogen": "कोई रोगज़नक़ नहीं पाया गया",
            "cause": "कोई फंगल संक्रमण या बीमारी के लक्षण नहीं मिले हैं। पत्ती में प्राकृतिक हरा क्लोरोफिल और स्वस्थ कोशिका संरचना है।",
            "symptoms": "चिकनी, एकसमान हरी पत्तियां। कोई धब्बे, सड़न या फफोले नहीं हैं।",
            "prevention": "1. संतुलित NPK (120:60:40 किग्रा/हेक्टेयर) उर्वरक दें।\n2. पहले 35 दिनों में समय पर खरपतवार नियंत्रण करें।\n3. फूल आते और दाना भरते समय पर्याप्त नमी रखें।\n4. फॉल आर्मीवर्म के लिए नियमित निरीक्षण करें।",
            "treatment": "किसी रासायनिक कवकनाशी की आवश्यकता नहीं है।",
            "dosage": "• सुरक्षात्मक जैविक देखभाल: स्यूडोमोनास @ 2 ग्राम/लीटर या नीम का तेल @ 3 मिली/लीटर का छिड़काव करके प्रतिरोधक क्षमता बनाए रखें।",
            "timing": "सामान्य मौसमी कृषि दिनचर्या का पालन करें।",
            "yield": "फसल पूरी तरह स्वस्थ है और अधिकतम पैदावार (6-8 टन/हेक्टेयर) देने में सक्षम है।",
            "solution": "फसल बहुत अच्छी स्थिति में है! उचित सिंचाई और संतुलित खाद प्रबंधन जारी रखें।"
        },
        "calc": {
            "chemical_name": "Bio-Protective Neem Oil (1500 ppm)",
            "dose_per_liter_g": 3.0,
            "dose_per_16L_pump_g": 48.0,
            "water_per_acre_L": 200,
            "pumps_per_acre_16L": 12.5,
            "total_pack_needed_g": 600,
            "approx_cost_inr": 120
        }
    }
}

# =========================================================
# GRAD-CAM ATTENTION HEATMAP COMPUTATION
# =========================================================

def compute_gradcam_overlay(image_array, predicted_class_index, alpha=0.45):
    """
    Computes a Grad-CAM Class Activation Map on the final Conv2D layer (conv2d_3)
    and returns a Base64-encoded PNG data URI overlaying the input image.
    """
    try:
        if disease_model is None:
            return None

        # Find the conv2d_3 layer (or final Conv2D layer)
        conv_idx = None
        for i, layer in enumerate(disease_model.layers):
            if layer.name == "conv2d_3":
                conv_idx = i
                break
        if conv_idx is None:
            for i in range(len(disease_model.layers) - 1, -1, -1):
                if isinstance(disease_model.layers[i], tf.keras.layers.Conv2D):
                    conv_idx = i
                    break
        if conv_idx is None:
            return None

        # Prepare tensor
        input_tensor = tf.cast(tf.expand_dims(image_array, axis=0), tf.float32)
        if input_tensor.shape[1:3] != IMAGE_SIZE:
            input_tensor = tf.image.resize(input_tensor, IMAGE_SIZE)

        # Forward pass up to conv layer
        x = input_tensor
        for i in range(conv_idx + 1):
            x = disease_model.layers[i](x)

        # Forward pass to output with gradient tracking
        with tf.GradientTape() as tape:
            tape.watch(x)
            conv_output = x
            y = x
            for i in range(conv_idx + 1, len(disease_model.layers)):
                y = disease_model.layers[i](y)
            loss = y[:, predicted_class_index]

        grads = tape.gradient(loss, conv_output)
        if grads is None:
            return None

        # Pool gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        # Resize heatmap to match image dimensions
        orig_h, orig_w = image_array.shape[:2]
        hm_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        hm_norm = np.array(hm_img, dtype=np.float32) / 255.0

        # Apply Jet colormap
        cmap = mpl.colormaps["jet"]
        colored = (cmap(hm_norm)[:, :, :3] * 255).astype(np.uint8)

        # Alpha-blend with original image
        blended = (
            alpha * colored.astype(np.float32) +
            (1.0 - alpha) * image_array.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        # Encode to Base64 PNG
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    except Exception as e:
        print(f"[-] Grad-CAM computation error: {e}")
        return None

# =========================================================
# MAIZE GATE & DISEASE INFERENCE
# =========================================================

def load_uploaded_image(file):
    """Safely decodes an uploaded file into an RGB NumPy uint8 array."""
    try:
        file.seek(0)
        image_bytes = file.read()
        if not image_bytes:
            return None
        image_tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        return np.clip(image_tensor.numpy(), 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[-] Image read error: {e}")
        return None

def check_maize_gate(image_array, lang="en"):
    """Verifies that the uploaded leaf is maize before disease classification."""
    rejection_messages = {
        "en": "This image does not appear to be a maize leaf. Please upload a clear maize crop image.",
        "kn": "ಈ ಚಿತ್ರವು ಮೆಕ್ಕೆಜೋಳದ ಎಲೆಯಂತೆ ಕಂಡುಬರುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಮೆಕ್ಕೆಜೋಳದ ಬೆಳೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "hi": "यह तस्वीर मक्के की पत्ती की प्रतीत नहीं होती है। कृपया मक्के की फसल की स्पष्ट तस्वीर अपलोड करें।"
    }

    if maize_gate is None:
        return {"accepted": False, "is_maize": False, "maize_confidence": 0.0, "message": "Gate model unavailable."}

    img = tf.image.resize(image_array.astype(np.float32), IMAGE_SIZE)
    gate_pred = maize_gate.predict(tf.expand_dims(img, axis=0), verbose=0)
    raw_val = float(np.squeeze(gate_pred))
    maize_conf = float(np.clip(1.0 - raw_val, 0.0, 1.0)) * 100.0

    if maize_conf < (MAIZE_GATE_THRESHOLD * 100.0):
        return {
            "accepted": False,
            "is_maize": False,
            "maize_confidence": round(maize_conf, 2),
            "message": rejection_messages.get(lang, rejection_messages["en"])
        }

    return {"accepted": True, "is_maize": True, "maize_confidence": round(maize_conf, 2), "message": "Valid maize leaf."}

def predict_disease(image_array, lang="en"):
    """Runs the multi-class convolutional disease classifier and generates Grad-CAM."""
    if disease_model is None:
        return {"success": False, "message": "Disease classification model unavailable."}

    img = tf.image.resize(image_array.astype(np.float32), IMAGE_SIZE)
    preds = disease_model.predict(tf.expand_dims(img, axis=0), verbose=0)
    probs = np.squeeze(preds)

    if probs.ndim == 0:
        class_idx = 1 if probs >= 0.5 else 0
        confidence = float(probs if class_idx == 1 else 1.0 - probs) * 100.0
    else:
        probs = probs.astype(np.float64)
        if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(np.sum(probs), 1.0, atol=0.01):
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / np.sum(exp_p)
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100.0

    if class_idx >= len(classes):
        class_idx = 0

    class_name = classes[class_idx]
    disease_entry = DISEASE_DATABASE.get(class_name, DISEASE_DATABASE["healthy"])
    lang_key = lang if lang in ["en", "kn", "hi"] else "en"
    info = disease_entry.get(lang_key, disease_entry["en"])
    calc_data = disease_entry.get("calc", {})

    heatmap_b64 = compute_gradcam_overlay(image_array, class_idx)

    return {
        "success": True,
        "disease": class_name,
        "disease_name": info["name"],
        "confidence": round(confidence, 2),
        "pathogen": info.get("pathogen", ""),
        "cause": info.get("cause", ""),
        "symptoms": info.get("symptoms", ""),
        "prevention": info.get("prevention", ""),
        "treatment": info.get("treatment", ""),
        "dosage": info.get("dosage", ""),
        "timing": info.get("timing", ""),
        "solution": info.get("solution", ""),
        "yield": info.get("yield", ""),
        "calc": calc_data,
        "multilingual": {
            "en": disease_entry.get("en", {}),
            "kn": disease_entry.get("kn", {}),
            "hi": disease_entry.get("hi", {})
        },
        "heatmap_image": heatmap_b64
    }

# =========================================================
# FLASK WEB ENDPOINTS
# =========================================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.route("/<path:filename>")
def static_files(filename):
    file_path = BASE_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(BASE_DIR, filename)
    return jsonify({"success": False, "message": "File not found."}), 404

@app.route("/predict", methods=["POST"])
def predict():
    lang = request.form.get("language") or request.args.get("language") or "en"
    lang = str(lang).strip().lower()

    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"success": False, "is_maize": False, "message": "No valid image file uploaded."}), 400

    image_array = load_uploaded_image(request.files["file"])
    if image_array is None:
        return jsonify({"success": False, "is_maize": False, "message": "Unable to decode image."}), 400

    # Step 1: Maize Verification Gate
    gate_result = check_maize_gate(image_array, lang=lang)
    if not gate_result["accepted"]:
        return jsonify({
            "success": False,
            "is_maize": False,
            "maize_confidence": gate_result["maize_confidence"],
            "message": gate_result["message"]
        }), 200

    # Step 2: Multi-class Disease Diagnosis + Grad-CAM Heatmap
    disease_result = predict_disease(image_array, lang=lang)
    if not disease_result["success"]:
        return jsonify({"success": False, "is_maize": True, "message": disease_result["message"]}), 500

    return jsonify({
        "success": True,
        "is_maize": True,
        "maize_confidence": gate_result["maize_confidence"],
        "disease": disease_result["disease"],
        "disease_name": disease_result["disease_name"],
        "confidence": disease_result["confidence"],
        "pathogen": disease_result["pathogen"],
        "cause": disease_result["cause"],
        "symptoms": disease_result["symptoms"],
        "prevention": disease_result["prevention"],
        "treatment": disease_result["treatment"],
        "dosage": disease_result["dosage"],
        "timing": disease_result["timing"],
        "solution": disease_result["solution"],
        "yield": disease_result["yield"],
        "calc": disease_result["calc"],
        "multilingual": disease_result["multilingual"],
        "heatmap_image": disease_result["heatmap_image"],
        "message": "Maize disease diagnosis completed successfully."
    })

# =========================================================
# MULTILINGUAL AI AGRI COPILOT CHATBOT
# =========================================================

def generate_bot_response(user_msg, lang="en"):
    msg = user_msg.lower().strip()

    # 1. Fungicide Dosage & Pump Queries
    if any(w in msg for w in ["dosage", "dose", "spray", "pump", "knapsack", "fungicide", "medicine", "chemical", "quantity", "ಔಷಧ", "ಪ್ರಮಾಣ", "ಪಂಪ್", "दवा", "मात्रा", "पंप", "कवकनाशी"]):
        if any(w in msg for w in ["rust", "ರಸ್ಟ್", "रतुआ"]):
            if lang == "kn":
                return "🌽 **ಕಾಮನ್ ರಸ್ಟ್ (ತುಕ್ಕು ರೋಗ) ಔಷಧ ಪ್ರಮಾಣ:**\n• **16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ**: 40 ಗ್ರಾಂ ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP ಅಥವಾ 16 ಮಿ.ಲೀ ಪ್ರಾಪಿಕೊನಜೋಲ್ (ಟಿಲ್ಟ್).\n• **ಪ್ರತಿ ಎಕರೆಗೆ**: 500 ಗ್ರಾಂ ಮ್ಯಾಂಕೋಜೆಬ್ 200 ಲೀಟರ್ ನೀರಿಗೆ (ಸುಮಾರು 12.5 ಪಂಪ್‌ಗಳು).\n• **ಜೈವಿಕ ಆಯ್ಕೆ**: ಬೇವಿನ ಎಣ್ಣೆ (1500 ppm) @ 16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ 50 ಮಿ.ಲೀ.\n⏱️ **ಸಿಂಪರಣಾ ಸಮಯ**: ಬೆಳಿಗ್ಗೆ 06:30 ರಿಂದ 09:30 ರೊಳಗೆ ಸಿಂಪಡಿಸಿ."
            elif lang == "hi":
                return "🌽 **कॉमन रस्ट (रतुआ) दवा व मात्रा:**\n• **16 लीटर स्प्रे पंप के लिए**: 40 ग्राम मैंकोजेब 75% WP या 16 मिली प्रोपिकोनाज़ोल (टिल्ट)।\n• **प्रति एकड़**: 500 ग्राम मैंकोजेब 200 लीटर पानी में (लगभग 12-13 पंप)।\n• **जैविक विकल्प**: नीम का तेल (1500 ppm) @ 50 मिली प्रति 16L पंप।\n⏱️ **छिड़काव समय**: सुबह 6:30 से 9:30 के बीच शांत हवा में छिड़कें।"
            return "🌽 **Common Rust Recommended Dosage:**\n• **For 16L Knapsack Pump**: 40g Mancozeb 75% WP OR 16mL Propiconazole 25% EC (Tilt).\n• **Per Acre**: 500g Mancozeb in 200L water (~12.5 pumps).\n• **Organic Option**: Neem oil 1500 ppm @ 50mL per 16L pump.\n⏱️ Spray early morning (06:30 AM – 09:30 AM)."

        if any(w in msg for w in ["blight", "northern", "ಅಂಗಮಾರಿ", "झुलसा"]):
            if lang == "kn":
                return "🌽 **ಉತ್ತರ ಎಲೆ ಅಂಗಮಾರಿ (NLB) ಔಷಧ ಪ್ರಮಾಣ:**\n• **16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ**: 16 ಮಿ.ಲೀ ಪ್ರಾಪಿಕೊನಜೋಲ್ 25% EC (ಟಿಲ್ಟ್) ಅಥವಾ 40 ಗ್ರಾಂ ಮ್ಯಾಂಕೋಜೆಬ್.\n• **ಪ್ರತಿ ಎಕರೆಗೆ**: 200 ಮಿ.ಲೀ ಟಿಲ್ಟ್ 200 ಲೀಟರ್ ನೀರಿಗೆ.\n• **ಜೈವಿಕ**: ಸೂಡೊಮೊನಾಸ್ @ 80 ಗ್ರಾಂ ಪ್ರತಿ 16L ಪಂಪ್‌ಗೆ.\n⏱️ ಹೂವಾಡುವ ಮುನ್ನ ಸಿಗಾರ್ ಆಕಾರದ ಕಲೆಗಳು ಕಂಡ ತಕ್ಷಣ ಸಿಂಪಡಿಸಿ."
            elif lang == "hi":
                return "🌽 **उत्तरी पत्ती झुलसा (NLB) दवा व मात्रा:**\n• **16 लीटर स्प्रे पंप के लिए**: 16 मिली प्रोपिकोनाज़ोल 25% EC (टिल्ट) या 40 ग्राम मैंकोजेब।\n• **प्रति एकड़**: 200 मिली टिल्ट 200 लीटर पानी में।\n• **जैविक**: स्यूडोमोनास @ 80 ग्राम प्रति 16L पंप।\n⏱️ भुट्टा बनने से पहले लंबे धब्बे दिखते ही सुबह छिड़कें।"
            return "🌽 **Northern Leaf Blight Recommended Dosage:**\n• **For 16L Knapsack Pump**: 16mL Propiconazole 25% EC (Tilt) OR 40g Mancozeb 75% WP.\n• **Per Acre**: 200mL Tilt in 200L water (~12.5 pumps).\n⏱️ Spray immediately when cigar-shaped lesions appear on middle leaves before silking."

        if any(w in msg for w in ["gray", "grey", "spot", "ಚುಕ್ಕೆ", "धब्बे"]):
            if lang == "kn":
                return "🌽 **ಗ್ರೇ ಲೀಫ್ ಸ್ಪಾಟ್ ಔಷಧ ಪ್ರಮಾಣ:**\n• **16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ**: 16 ಗ್ರಾಂ ಪೈರಾಕ್ಲೋಸ್ಟ್ರೋಬಿನ್ 20% WG ಅಥವಾ 32 ಗ್ರಾಂ ಸಾಫ್ (ಕಾರ್ಬೆಂಡಾಜಿಮ್ + ಮ್ಯಾಂಕೋಜೆಬ್).\n• **ಪ್ರತಿ ಎಕರೆಗೆ**: 200 ಗ್ರಾಂ ಪೈರಾಕ್ಲೋಸ್ಟ್ರೋಬಿನ್ 200 ಲೀಟರ್ ನೀರಿಗೆ.\n⏱️ ತೆನೆ ಎಲೆಯ ಕೆಳಗೆ ಚೌಕಾಕಾರದ ಕಲೆಗಳು ಕಂಡಾಗ ಬೆಳಿಗ್ಗೆ ಸಿಂಪಡಿಸಿ."
            elif lang == "hi":
                return "🌽 **ग्रे लीफ स्पॉट दवा व मात्रा:**\n• **16 लीटर स्प्रे पंप के लिए**: 16 ग्राम पायराक्लोस्ट्रोबिन 20% WG या 32 ग्राम साफ कवकनाशी।\n• **प्रति एकड़**: 200 ग्राम पायराक्लोस्ट्रोबिन 200 लीटर पानी में।\n⏱️ भुट्टे की पत्ती के नीचे आयताकार धब्बे दिखते ही छिड़काव करें।"
            return "🌽 **Gray Leaf Spot Recommended Dosage:**\n• **For 16L Knapsack Pump**: 16g Pyraclostrobin 20% WG OR 32g Saaf.\n• **Per Acre**: 200g Pyraclostrobin in 200L water.\n⏱️ Apply between V12 and silking stage."

        # General dosage
        if lang == "kn":
            return "🌽 **ಸಾಮಾನ್ಯ ಶಿಲೀಂಧ್ರನಾಶಕ ಪ್ರಮಾಣ (16 ಲೀಟರ್ ಪಂಪ್‌ಗೆ):**\n• ರಕ್ಷಣಾತ್ಮಕ: ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP @ 40 ಗ್ರಾಂ.\n• ಗುಣಪಡಿಸುವ: ಪ್ರಾಪಿಕೊನಜೋಲ್ (ಟಿಲ್ಟ್) @ 16 ಮಿ.ಲೀ.\n• ಎಕರೆಗೆ 200 ಲೀಟರ್ ನೀರು (ಸುಮಾರು 12.5 ಪಂಪ್‌ಗಳು) ಅಗತ್ಯ."
        elif lang == "hi":
            return "🌽 **सामान्य कवकनाशी मात्रा (16 लीटर पंप के लिए):**\n• सुरक्षात्मक: मैंकोजेब 75% WP @ 40 ग्राम।\n• उपचारात्मक: प्रोपिकोनाज़ोल (टिल्ट) @ 16 मिली।\n• प्रति एकड़ 200 लीटर पानी (लगभग 12-13 पंप) का छिड़काव करें।"
        return "🌽 **Standard Knapsack Pump Dosage (16 Liters):**\n• Protective: Mancozeb 75% WP @ 40g per 16L pump.\n• Curative Systemic: Propiconazole 25% EC (Tilt) @ 16mL per 16L pump.\n• One acre requires ~200L water (12.5 knapsack pumps)."

    # 2. Disease Causes
    if any(w in msg for w in ["cause", "why", "pathogen", "reason", "spread", "ಕಾರಣ", "ಏಕೆ", "कारण", "क्यो"]):
        if any(w in msg for w in ["rust", "ರಸ್ಟ್", "रतुआ"]):
            if lang == "kn":
                return "🍄 **ಕಾಮನ್ ರಸ್ಟ್ ಕಾರಣ:** ಇದು *Puccinia sorghi* ಶಿಲೀಂಧ್ರದಿಂದ ಬರುತ್ತದೆ. ತಂಪಾದ ಹವಾಮಾನ (16°C–25°C), ಅಧಿಕ ತೇವಾಂಶ (>95%) ಮತ್ತು ಎಲೆಗಳ ಮೇಲೆ 6 ಗಂಟೆಗಿಂತ ಹೆಚ್ಚು ಕಾಲ ಇಬ್ಬನಿ ನಿಲ್ಲುವುದರಿಂದ ಗಾಳಿಯ ಮೂಲಕ ಹರಡುತ್ತದೆ."
            elif lang == "hi":
                return "🍄 **कॉमन रस्ट का कारण:** यह *Puccinia sorghi* कवक से होता है। 16°C–25°C तापमान, उच्च आर्द्रता और पत्तियों पर 6+ घंटे तक ओस रहने से यह हवा द्वारा फैलता है।"
            return "🍄 **Cause of Common Rust:** Caused by the fungus *Puccinia sorghi*. Airborne spores spread under moderate temperatures (16°C–25°C), high humidity (>95%), and 6+ hours of leaf wetness."

        if any(w in msg for w in ["blight", "northern", "ಅಂಗಮಾರಿ", "झुलसा"]):
            if lang == "kn":
                return "🍄 **ಉತ್ತರ ಎಲೆ ಅಂಗಮಾರಿ ಕಾರಣ:** ಇದು *Exserohilum turcicum* ಶಿಲೀಂಧ್ರದಿಂದ ಬರುತ್ತದೆ. ಹಳೆಯ ಬೆಳೆ ಕಸದಲ್ಲಿ ಉಳಿಯುವ ಬೀಜಕಗಳು ಮಳೆ ಹನಿ ಮತ್ತು ಗಾಳಿಯಿಂದ ಹರಡುತ್ತವೆ (18°C–27°C ತಾಪಮಾನ)."
            elif lang == "hi":
                return "🍄 **उत्तरी पत्ती झुलसा का कारण:** यह *Exserohilum turcicum* फंगस से होता है। पुराने फसल अवशेषों में यह जीवित रहता है और बारिश व हवा से नई पत्तियों पर फैलता है।"
            return "🍄 **Cause of Northern Leaf Blight:** Caused by *Exserohilum turcicum*. It overwinters in crop debris and spreads via rain-splash and wind during moderate temperatures (18°C–27°C)."

        if any(w in msg for w in ["gray", "grey", "spot", "ಚುಕ್ಕೆ", "धब्बे"]):
            if lang == "kn":
                return "🍄 **ಗ್ರೇ ಲೀಫ್ ಸ್ಪಾಟ್ ಕಾರಣ:** ಇದು *Cercospora zeae-maydis* ಶಿಲೀಂಧ್ರದಿಂದ ಬರುತ್ತದೆ. ಬೆಚ್ಚನೆಯ ತಾಪಮಾನ (25°C–32°C), ನಿರಂತರ ಮೋಡ ಮತ್ತು ಸತತ ಮೆಕ್ಕೆಜೋಳ ಬೆಳೆಯುವುದರಿಂದ ಇದು ಉಂಟಾಗುತ್ತದೆ."
            elif lang == "hi":
                return "🍄 **ग्रे लीफ स्पॉट का कारण:** यह *Cercospora zeae-maydis* कवक से होता है। गर्म तापमान (25°C–32°C), उच्च नमी और लगातार मक्के की खेती से यह फैलता है।"
            return "🍄 **Cause of Gray Leaf Spot:** Caused by *Cercospora zeae-maydis*. Favored by warm temperatures (25°C–32°C), high humidity, and continuous maize monoculture."

    # 3. Prevention Practices
    if any(w in msg for w in ["prevent", "avoid", "protect", "control", "ತಡೆಗಟ್ಟು", "ನಿಯಂತ್ರಣ", "रोकथाम", "बचाव"]):
        if lang == "kn":
            return "🛡️ **ಮೆಕ್ಕೆಜೋಳ ರೋಗ ತಡೆಗಟ್ಟುವ ಕ್ರಮಗಳು:**\n1. ರೋಗ ನಿರೋಧಕ ಹೈಬ್ರಿಡ್ ಬೀಜ ಬಳಸಿ.\n2. ದ್ವಿದಳ ಧಾನ್ಯಗಳೊಂದಿಗೆ (ಸೋಯಾಬೀನ್/ಕಡಲೆಕಾಯಿ) ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ.\n3. ಸಾಲು ಅಂತರ 60 ಸೆಂ.ಮೀ, ಗಿಡದ ಅಂತರ 20-25 ಸೆಂ.ಮೀ ಕಾಪಾಡಿ.\n4. ಸಂಜೆ ವೇಳೆ ನೀರು ಹಾಯಿಸಬೇಡಿ; ಹೊಲದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಕಾಲುವೆ ಮಾಡಿ.\n5. ಕೊಯ್ಲಿನ ನಂತರ ಹಳೆಯ ಬೆಳೆ ಕಸವನ್ನು ಆಳವಾಗಿ ಉಳುಮೆ ಮಾಡಿ."
        elif lang == "hi":
            return "🛡️ **मक्का रोग रोकथाम के उपाय:**\n1. प्रमाणित रोगरोधी संकर (Hybrid) बीजों का चयन करें।\n2. दलहनी फसलों के साथ 1-2 वर्ष का फसल चक्र अपनाएं।\n3. कतार से कतार 60 सेमी और पौधे से पौधे 20 सेमी की दूरी रखें।\n4. शाम के समय फव्वारा सिंचाई से बचें; जल निकासी अच्छी रखें।\n5. कटाई के बाद खेत की गहरी जुताई करके पुराने अवशेषों को नष्ट करें।"
        return "🛡️ **Maize Disease Prevention Protocols:**\n1. Plant certified disease-resistant hybrids.\n2. Rotate fields with non-host crops (soybean, pulses) for 1–2 seasons.\n3. Maintain 60 cm row spacing and 20–25 cm plant spacing for canopy aeration.\n4. Avoid late evening overhead irrigation to limit leaf wetness.\n5. Deep plow crop residues after harvest to accelerate decomposition."

    # 4. Fertilizers & Nutrition
    if any(w in msg for w in ["fertiliz", "urea", "npk", "dap", "zinc", "ಗೊಬ್ಬರ", "ಖಾತ್", "खाद", "उर्वरक"]):
        if lang == "kn":
            return "🌱 **ರಸಗೊಬ್ಬರ ವೇಳಾಪಟ್ಟಿ (NPK 120:60:40 kg/ha):**\n• ಬಿತ್ತನೆ ಸಮಯದಲ್ಲಿ: ಪೂರ್ಣ DAP, ಪೂರ್ಣ ಪೊಟ್ಯಾಶ್, ಮತ್ತು 1/3 ಯೂರಿಯಾ.\n• 30-35 ದಿನಕ್ಕೆ: 1/3 ಯೂರಿಯಾ ಮೇಲುಗೊಬ್ಬರವಾಗಿ ನೀಡಿ.\n• 50-55 ದಿನಕ್ಕೆ (ಹೂವಾಡುವ ಹಂತ): ಉಳಿದ 1/3 ಯೂರಿಯಾ ನೀಡಿ.\n• ಜಿಂಕ್ ಕೊರತೆಗೆ: ಜಿಂಕ್ ಸಲ್ಫೇಟ್ 25 ಕೆಜಿ/ಹೆಕ್ಟೇರ್ ಮಣ್ಣಿಗೆ ಸೇರಿಸಿ."
        elif lang == "hi":
            return "🌱 **संतुलित खाद अनुसूची (NPK 120:60:40 किग्रा/हेक्टेयर):**\n• बुवाई के समय: पूरा DAP, पूरा पोटाश व 1/3 यूरिया डालें।\n• 30-35 दिन (घुटने की ऊंचाई): 1/3 यूरिया की टॉप-ड्रेसिंग करें।\n• 50-55 दिन (फूल/भुट्टा आते समय): बाकी 1/3 यूरिया डालें।\n• जिंक की कमी पर 25 किग्रा जिंक सल्फेट प्रति हेक्टेयर दें।"
        return "🌱 **Fertilizer Protocol (NPK 120:60:40 kg/ha):**\n• Basal at sowing: 100% Phosphorus (DAP), 100% Potassium (MOP), and 33% Nitrogen (Urea).\n• First top-dress (30–35 days): 33% Nitrogen at knee-high stage.\n• Second top-dress (50–55 days): Remaining 34% Nitrogen just prior to tasseling.\n• Micronutrients: Apply Zinc Sulfate @ 25 kg/ha basally."

    # 5. Fall Armyworm / Pests
    if any(w in msg for w in ["worm", "pest", "armyworm", "insect", "ಹುಳು", "ಕೀಟ", "कीट", "इल्ली"]):
        if lang == "kn":
            return "🐛 **ಸೈನಿಕ ಹುಳು (Fall Armyworm) ನಿಯಂತ್ರಣ:** ಬೇವಿನ ಎಣ್ಣೆ (1500 ppm) @ 16L ಪಂಪ್‌ಗೆ 50 ಮಿ.ಲೀ ಅಥವಾ ಎಮಾಮೆಕ್ಟಿನ್ ಬೆಂಜೊಯೆಟ್ 5% SG @ 16L ಪಂಪ್‌ಗೆ 7 ಗ್ರಾಂ ಸುಳಿಯೊಳಗೆ ಬೀಳುವಂತೆ ಸಿಂಪಡಿಸಿ."
        elif lang == "hi":
            return "🐛 **फॉल आर्मीवर्म नियंत्रण:** नीम का तेल (1500 ppm) @ 50 मिली प्रति 16L पंप या इमामेक्टिन बेंजोएट 5% SG @ 7 ग्राम प्रति 16L पंप का पौधों की गोभ में छिड़काव करें।"
        return "🐛 **Fall Armyworm Management:** Apply Neem Oil (1500 ppm) @ 50mL per 16L pump or Emamectin Benzoate 5% SG @ 7g per 16L pump directly into plant whorls."

    # 6. Yield & Spacing
    if any(w in msg for w in ["yield", "water", "irrigation", "spacing", "ಇಳುವರಿ", "ನೀರಾವರಿ", "पैदावार", "सिंचाई"]):
        if lang == "kn":
            return "🌾 **ಇಳುವರಿ ಹೆಚ್ಚಿಸಲು ಸಲಹೆಗಳು:** ಸಾಲು ಅಂತರ 60 ಸೆಂ.ಮೀ, ಗಿಡದ ಅಂತರ 20-25 ಸೆಂ.ಮೀ ಇರಲಿ. ಹೂವು ಬಿಡುವ ಮತ್ತು ಕಾಳು ತುಂಬುವ ಹಂತಗಳಲ್ಲಿ ಸಮರ್ಪಕ ನೀರಾವರಿ ನೀಡಿ. ಮೊದಲ 35 ದಿನಗಳಲ್ಲಿ ಕಳೆ ನಿಯಂತ್ರಿಸಿ."
        elif lang == "hi":
            return "🌾 **उपज बढ़ाने के उपाय:** कतार से कतार 60 सेमी और पौधे से पौधे 20 सेमी की दूरी रखें। फूल आते और दाना भरते समय समय पर सिंचाई करें। पहले 35 दिनों में खरपतवार मुक्त रखें।"
        return "🌾 **Maximizing Yield:** Maintain 60 cm row and 20–25 cm plant spacing (~65,000 plants/ha). Provide critical irrigation at flowering (silking) and grain filling. Weed within first 35 days."

    # 7. APMC Mandi Rates & MSP
    if any(w in msg for w in ["mandi", "price", "rate", "msp", "market", "ಮಾರುಕಟ್ಟೆ", "ಬೆಲೆ", "ಮಂಡಿ", "ದರ", "मंडी", "भाव", "दाम", "मूल्य"]):
        if lang == "kn":
            return "💰 **ಮೆಕ್ಕೆಜೋಳ APMC ಮಾರುಕಟ್ಟೆ ದರ & MSP:**\n• ಕೇಂದ್ರ ಸರ್ಕಾರದ ಕನಿಷ್ಠ ಬೆಂಬಲ ಬೆಲೆ (MSP): ₹2,225/ಕ್ವಿಂಟಾಲ್.\n• ರಾಜ್ಯದ APMC ಮಾರುಕಟ್ಟೆ ದರ (ದಾವಣಗೆರೆ, ರಾಣೆಬೆನ್ನೂರು): ₹2,250 ರಿಂದ ₹2,420/ಕ್ವಿಂಟಾಲ್.\n• ಗರಿಷ್ಠ ಬೆಲೆ ಪಡೆಯಲು ಕಾಳಿನ ತೇವಾಂಶ 14% ಕ್ಕಿಂತ ಕಡಿಮೆ ಇರುವಂತೆ ಒಣಗಿಸಿ ಮಂಡಿಗೆ ತರುವುದು ಕಡ್ಡಾಯ."
        elif lang == "hi":
            return "💰 **मक्का APMC मंडी भाव एवं MSP:**\n• सरकारी न्यूनतम समर्थन मूल्य (MSP): ₹2,225 प्रति क्विंटल।\n• प्रमुख मंडियों में मॉडल भाव: ₹2,250 से ₹2,420 प्रति क्विंटल।\n• पूरा भाव पाने के लिए दाने में नमी 14% से कम रखें और कचरा साफ करके मंडी ले जाएं।"
        return "💰 **Maize APMC Market Rates & MSP:**\n• Government Minimum Support Price (MSP): ₹2,225 / Quintal.\n• Key APMC Mandi Modal Rates: ₹2,250 – ₹2,420 / Quintal depending on grain quality.\n• Farmer Tip: Ensure grain moisture is below 14% to avoid price deductions at mandi weighbridges."

    # 8. Kisan Call Center & Helpline
    if any(w in msg for w in ["call", "helpline", "toll", "phone", "contact", "support", "help", "ಸಹಾಯವಾಣಿ", "ಹೆಲ್ಪ್‌ಲೈನ್", "ಫೋನ್", "ಸಂಪರ್ಕ", "हेल्पलाइन", "फोन", "संपर्क", "नंबर"]):
        if lang == "kn":
            return "📞 **ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್ ಮತ್ತು ಕೃಷಿ ಸಹಾಯವಾಣಿ:**\n• ಟೋಲ್-ಫ್ರೀ ಸಂಖ್ಯೆ: 1800-180-1551 (ಬೆಳಗ್ಗೆ 6:00 ರಿಂದ ರಾತ್ರಿ 10:00 ವರೆಗೆ ಉಚಿತ).\n• ಕನ್ನಡದಲ್ಲೇ ಕೃಷಿ ತಜ್ಞರೊಂದಿಗೆ ಮಾತನಾಡಿ ಬೀಜ, ರೋಗ ಹಾಗೂ ಬೆಳೆ ವಿಮೆಯ ಬಗ್ಗೆ ನೇರ ಸಲಹೆ ಪಡೆಯಬಹುದು."
        elif lang == "hi":
            return "📞 **किसान कॉल सेंटर एवं कृषि हेल्पलाइन:**\n• टोल-फ्री नंबर: 1800-180-1551 (सुबह 6:00 से रात 10:00 बजे तक निःशुल्क)।\n• अपनी मातृभाषा (हिन्दी/कन्नड़) में सरकारी कृषि वैज्ञानिकों से सीधे बात करें और योजनाओं की जानकारी लें।"
        return "📞 **Kisan Call Center & Government Helpline:**\n• Toll-Free National Hotline: 1800-180-1551 (Available 6:00 AM – 10:00 PM, 7 days/week).\n• Connects you directly to certified government agronomists in your local language (Kannada, Hindi, English)."

    # 9. Spray Safety, PPE & Pre-Harvest Interval (PHI)
    if any(w in msg for w in ["safe", "safety", "phi", "harvest", "ppe", "mask", "wind", "spray time", "weather", "ಸುರಕ್ಷತೆ", "ಕೊಯ್ಲು", "ಗಾಳಿ", "ರಕ್ಷಣೆ", "सुरक्षा", "कटाई", "मास्क", "हवा"]):
        if lang == "kn":
            return "🛡️ **ರಾಸಾಯನಿಕ ಸಿಂಪರಣೆ ಸುರಕ್ಷತೆ ಮತ್ತು PHI ನಿಯಮ:**\n• PHI (ಕೊಯ್ಲು ಪೂರ್ವ ಅಂತರ): ಕೊಯ್ಲಿಗೆ 14 ದಿನ ಮುಂಚಿತವಾಗಿ ರಾಸಾಯನಿಕ ಸಿಂಪರಣೆ ನಿಲ್ಲಿಸಿ.\n• ಮುನ್ನೆಚ್ಚರಿಕೆ: ಬಾಯಿಗೆ ಮಾಸ್ಕ್, ಕೈಗೆ ಗ್ಲೌಸ್ ಧರಿಸಿ. ಗಾಳಿ ಬೀಸುವ ದಿಕ್ಕಿನಲ್ಲೇ ಸಿಂಪಡಿಸಿ (ಗಾಳಿಗೆ ಎದುರಾಗಿ ಬೇಡ).\n• ಸೂಕ್ತ ಸಮಯ: ಮುಂಜಾನೆ 6:30 ರಿಂದ 9:30 ರೊಳಗೆ ಸಿಂಪಡಿಸಿ; ತೀವ್ರ ಬಿಸಿಲು ಹಾಗೂ ಮಳೆಯ ಸಂಭವವಿದ್ದಾಗ ಸಿಂಪಡಿಸಬೇಡಿ."
        elif lang == "hi":
            return "🛡️ **कीटनाशक छिड़काव सुरक्षा एवं PHI नियम:**\n• PHI (कटाई पूर्व अंतराल): भुट्टे तोड़ने/कटाई से कम से कम 14 दिन पहले फफूंदनाशक का छिड़काव बंद कर दें।\n• सुरक्षा उपाय: मास्क और रबर के दस्ताने पहनें। कभी भी हवा की विपरीत दिशा में स्प्रे न करें।\n• सर्वोत्तम समय: सुबह 6:30 से 9:30 बजे तक। तेज धूप या बारिश की संभावना में छिड़काव न करें।"
        return "🛡️ **Pesticide Safety & Pre-Harvest Interval (PHI):**\n• Pre-Harvest Interval (PHI): Stop all chemical fungicide applications at least 14 days before harvesting cobs for consumer safety.\n• PPE Checklist: Wear N95 face mask, eye protection, and nitrile gloves. Always spray in the direction of the wind (never against it).\n• Spray Window: Early morning (6:30 AM – 9:30 AM) when wind speeds are <10 km/h and canopy is dry."

    # General Fallback
    if lang == "kn":
        return "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಮಿತ್ರ AI ಸಹಾಯಕ. ಮೆಕ್ಕೆಜೋಳದ ರೋಗಗಳು, ಕಾರಣಗಳು, ತಡೆಗಟ್ಟುವಿಕೆ, 16 ಲೀಟರ್ ಪಂಪ್ ಔಷಧ ಪ್ರಮಾಣ, ಮಂಡಿ ಬೆಲೆ ಹಾಗೂ ರಸಗೊಬ್ಬರದ ಬಗ್ಗೆ ಕೇಳಿ!"
    elif lang == "hi":
        return "नमस्ते! मैं आपका कृषि मित्र AI सहायक हूँ। मक्के के रोगों, उनके कारण, रोकथाम, 16 लीटर स्प्रे पंप की दवा की मात्रा, मंडी भाव और खाद के बारे में पूछें!"
    return "Hello! I am your Krishi Mitra AI Copilot. Ask me about maize disease causes, prevention, exact 16L knapsack sprayer dosages, APMC mandi rates, Kisan helpline, or fertilizers!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    language = str(data.get("language", "en")).strip().lower()

    if not message:
        return jsonify({"success": False, "reply": "Please provide a question."}), 400

    reply = generate_bot_response(message, language)
    return jsonify({"success": True, "reply": reply})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "app_name": "Krishi Mitra",
        "maize_gate_loaded": maize_gate is not None,
        "disease_model_loaded": disease_model is not None,
        "classes": classes,
        "maize_threshold": MAIZE_GATE_THRESHOLD * 100,
        "gate_model": GATE_MODEL_PATH.name,
        "disease_model": DISEASE_MODEL_PATH.name
    })

# =========================================================
# SERVER ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print(f"[+] Starting Krishi Mitra Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)