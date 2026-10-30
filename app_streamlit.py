import csv
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "olive_multiclass_v3.pth"
CLASS_KEYS = ["healthy", "aculus", "peacock"]  # must match training order: Healthy, aculus_olearius, olive_peacock_spot
LOW_CONF = 70.0  # percent

# Feedback is saved locally as feedback/<class_folder>/<image>.jpg + feedback/log.csv.
# Folder names match the training dataset, so verified images can be copied straight in.
FEEDBACK_DIR = Path("feedback")
DIR_NAMES = {"healthy": "Healthy", "aculus": "aculus_olearius", "peacock": "olive_peacock_spot"}

# Photo-quality thresholds. These are heuristics and are NOT tuned yet:
# open "Measured values" in the app on a few photos and adjust.
BLUR_MIN = 60.0      # variance of the Laplacian (lower = blurrier)
DARK_MAX = 60.0      # mean gray level below this = too dark
BRIGHT_MIN = 235.0   # mean gray level above this = overexposed
COVER_MIN = 0.04     # fraction of colorful pixels below this = leaf too small
COVER_MAX = 0.60     # above this = busy background or several leaves

TEXT = {
    "en": {
        "lang": "Language",
        "title": "Olive Leaf Disease Detector",
        "upload": "Upload an olive leaf photo",
        "pred": "Prediction",
        "conf": "Confidence",
        "probs": "Probabilities",
        "low": "Low confidence. Retake the photo: one leaf, plain light background, good light.",
        "warn": "This model was trained on studio-style photos (single leaf, plain background). Results on field photos can be wrong, even when the confidence is high.",
        "desc": "About",
        "treat": "Treatment (general info)",
        "prev": "Prevention",
        "advice": "General information only. Ask a local agronomist before treating.",
        "q_title": "Photo quality check",
        "q": {
            "blur": "The photo looks blurry.",
            "dark": "The photo is too dark.",
            "bright": "The photo is overexposed.",
            "small": "The leaf looks too small in the frame. Move closer.",
            "busy": "The background looks busy or there are several leaves. Use one leaf on a plain background.",
        },
        "fb_title": "Help improve the model",
        "fb_ask": "Is the prediction correct?",
        "fb_yes": "Yes",
        "fb_no": "No",
        "fb_true": "What is the correct class?",
        "fb_consent": "I allow this photo to be saved to improve the model.",
        "fb_save": "Save feedback",
        "fb_saved": "Saved. Thank you.",
        "fb_note": "Only save labels you are sure about.",
        "fb_area": "Did the heatmap highlight the right part of the leaf? (optional)",
        "fb_unsure": "Not sure",
        "cam_title": "Where the model looked",
        "cam_caption": "Red marks areas that influenced the prediction. It is a coarse map, not a diagnosis.",
        "names": {"healthy": "Healthy", "aculus": "Aculus olearius (olive bud mite)", "peacock": "Peacock spot"},
        "info": {
            "healthy": ("No disease signs detected.", "", ""),
            "aculus": (
                "Microscopic mite that causes leaf deformation, discoloration and bud damage.",
                "Sulfur-based or acaricide treatments at the right time. Ask a local agronomist for products and timing.",
                "Prune to improve air flow and light. Monitor young shoots and buds in spring.",
            ),
            "peacock": (
                "Fungal disease (Spilocaea oleagina) with dark circular spots, often with a yellow halo, leading to leaf drop.",
                "Copper-based fungicides applied before rainy periods. Ask a local agronomist for timing.",
                "Prune for air flow, remove fallen leaves, choose tolerant varieties, avoid long leaf wetness.",
            ),
        },
    },
    "fr": {
        "lang": "Langue",
        "title": "Détecteur de maladies des feuilles d'olivier",
        "upload": "Téléversez une photo de feuille d'olivier",
        "pred": "Prédiction",
        "conf": "Confiance",
        "probs": "Probabilités",
        "low": "Confiance faible. Reprenez la photo : une seule feuille, fond clair uni, bonne lumière.",
        "warn": "Ce modèle a été entraîné sur des photos de type studio (une feuille, fond uni). Les résultats sur des photos de terrain peuvent être faux, même avec une confiance élevée.",
        "desc": "À propos",
        "treat": "Traitement (information générale)",
        "prev": "Prévention",
        "advice": "Information générale uniquement. Demandez conseil à un agronome local avant de traiter.",
        "q_title": "Contrôle de la qualité de la photo",
        "q": {
            "blur": "La photo semble floue.",
            "dark": "La photo est trop sombre.",
            "bright": "La photo est surexposée.",
            "small": "La feuille semble trop petite dans le cadre. Rapprochez-vous.",
            "busy": "Le fond semble chargé ou il y a plusieurs feuilles. Utilisez une seule feuille sur un fond uni.",
        },
        "fb_title": "Aidez à améliorer le modèle",
        "fb_ask": "La prédiction est-elle correcte ?",
        "fb_yes": "Oui",
        "fb_no": "Non",
        "fb_true": "Quelle est la bonne classe ?",
        "fb_consent": "J'autorise l'enregistrement de cette photo pour améliorer le modèle.",
        "fb_save": "Enregistrer",
        "fb_saved": "Enregistré. Merci.",
        "fb_note": "N'enregistrez que des étiquettes dont vous êtes sûr(e).",
        "fb_area": "La carte de chaleur met-elle en évidence la bonne partie de la feuille ? (facultatif)",
        "fb_unsure": "Pas sûr(e)",
        "cam_title": "Où le modèle a regardé",
        "cam_caption": "Le rouge indique les zones qui ont influencé la prédiction. Carte approximative, pas un diagnostic.",
        "names": {"healthy": "Saine", "aculus": "Aculus olearius (phytopte de l'olivier)", "peacock": "Œil de paon"},
        "info": {
            "healthy": ("Aucun signe de maladie détecté.", "", ""),
            "aculus": (
                "Acarien microscopique qui provoque la déformation et la décoloration des feuilles et des dégâts sur les bourgeons.",
                "Traitements à base de soufre ou acaricides au bon moment. Demandez les produits et la période à un agronome local.",
                "Taillez pour améliorer l'aération et la lumière. Surveillez les jeunes pousses et les bourgeons au printemps.",
            ),
            "peacock": (
                "Maladie fongique (Spilocaea oleagina) avec des taches rondes sombres, souvent entourées d'un halo jaune, qui entraîne la chute des feuilles.",
                "Fongicides à base de cuivre appliqués avant les périodes pluvieuses. Demandez la période à un agronome local.",
                "Taillez pour aérer, ramassez les feuilles tombées, choisissez des variétés tolérantes, évitez l'humidité prolongée du feuillage.",
            ),
        },
    },
    "ar": {
        "lang": "اللغة",
        "title": "كاشف أمراض أوراق الزيتون",
        "upload": "ارفع صورة لورقة زيتون",
        "pred": "التشخيص",
        "conf": "درجة الثقة",
        "probs": "الاحتمالات",
        "low": "الثقة منخفضة. أعد التقاط الصورة: ورقة واحدة، خلفية فاتحة سادة، وإضاءة جيدة.",
        "warn": "تم تدريب هذا النموذج على صور بأسلوب الاستوديو (ورقة واحدة وخلفية سادة). قد تكون النتائج على صور الحقل خاطئة حتى مع ثقة عالية.",
        "desc": "نبذة",
        "treat": "العلاج (معلومات عامة)",
        "prev": "الوقاية",
        "advice": "معلومات عامة فقط. استشر مهندسًا زراعيًا محليًا قبل أي معالجة.",
        "q_title": "فحص جودة الصورة",
        "q": {
            "blur": "الصورة تبدو غير واضحة.",
            "dark": "الصورة مظلمة جدًا.",
            "bright": "الصورة مفرطة الإضاءة.",
            "small": "الورقة تبدو صغيرة جدًا في الإطار. اقترب أكثر.",
            "busy": "الخلفية تبدو مزدحمة أو توجد عدة أوراق. استخدم ورقة واحدة على خلفية سادة.",
        },
        "fb_title": "ساعد في تحسين النموذج",
        "fb_ask": "هل التشخيص صحيح؟",
        "fb_yes": "نعم",
        "fb_no": "لا",
        "fb_true": "ما هو التصنيف الصحيح؟",
        "fb_consent": "أسمح بحفظ هذه الصورة لتحسين النموذج.",
        "fb_save": "حفظ",
        "fb_saved": "تم الحفظ. شكرًا لك.",
        "fb_note": "احفظ فقط التصنيفات التي أنت متأكد منها.",
        "fb_area": "هل أبرزت الخريطة الحرارية الجزء الصحيح من الورقة؟ (اختياري)",
        "fb_unsure": "لست متأكدًا",
        "cam_title": "أين نظر النموذج",
        "cam_caption": "اللون الأحمر يشير إلى المناطق التي أثرت في التشخيص. خريطة تقريبية وليست تشخيصًا.",
        "names": {"healthy": "سليمة", "aculus": "حلم براعم الزيتون (Aculus olearius)", "peacock": "عين الطاووس"},
        "info": {
            "healthy": ("لم يتم اكتشاف أي علامات مرض.", "", ""),
            "aculus": (
                "حلم مجهري يسبب تشوه الأوراق وتغير لونها وإتلاف البراعم.",
                "معالجات بالكبريت أو مبيدات الحلم في الوقت المناسب. استشر مهندسًا زراعيًا لاختيار المنتج والتوقيت.",
                "قم بالتقليم لتحسين التهوية والإضاءة، وراقب الأغصان الفتية والبراعم في الربيع.",
            ),
            "peacock": (
                "مرض فطري (Spilocaea oleagina) يسبب بقعًا دائرية داكنة غالبًا بهالة صفراء وتساقط الأوراق.",
                "مبيدات فطرية نحاسية تُطبَّق قبل فترات الأمطار. استشر مهندسًا زراعيًا لتحديد التوقيت.",
                "التقليم لتحسين التهوية، وجمع الأوراق المتساقطة، واختيار أصناف متحملة، وتجنب بقاء الأوراق رطبة لفترات طويلة.",
            ),
        },
    },
}

LANG_LABELS = {"English": "en", "Français": "fr", "العربية": "ar"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model, img):
    with torch.no_grad():
        probs = torch.softmax(model(TRANSFORM(img).unsqueeze(0)), dim=1)[0]
    return [p.item() * 100 for p in probs]


def gradcam_overlay(model, img, class_idx):
    """Grad-CAM on ResNet18 layer4: red = areas that pushed toward class_idx."""
    x = TRANSFORM(img).unsqueeze(0)
    with torch.enable_grad():
        out = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        out = model.layer3(model.layer2(model.layer1(out)))
        feats = model.layer4(out)
        feats.retain_grad()
        logits = model.fc(torch.flatten(model.avgpool(feats), 1))
        model.zero_grad()
        logits[0, class_idx].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    disp = img.copy()
    disp.thumbnail((800, 800))
    heat = cv2.resize(cam, disp.size)  # disp.size is (width, height)
    base = np.array(disp).astype(np.float32)
    over = base.copy()
    over[..., 0] = np.clip(base[..., 0] + 255 * heat, 0, 255)
    over[..., 1] = base[..., 1] * (1 - 0.6 * heat)
    over[..., 2] = base[..., 2] * (1 - 0.6 * heat)
    return over.astype(np.uint8)


def save_feedback(img, true_key, predicted_key, confidence, original_name, area=""):
    folder = FEEDBACK_DIR / DIR_NAMES[true_key]
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.jpg"
    img.save(folder / name, quality=95)

    log = FEEDBACK_DIR / "log.csv"
    new_file = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["file", "true_class", "predicted_class", "confidence", "original_name", "heatmap_on_right_area"])
        writer.writerow([f"{DIR_NAMES[true_key]}/{name}", DIR_NAMES[true_key],
                         DIR_NAMES[predicted_key], f"{confidence:.1f}", original_name, area])


def quality_checks(img):
    """OpenCV input checks. Returns (measured values, list of warning keys)."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    # Leaf coverage: share of "colorful" pixels (Otsu threshold on saturation)
    sat = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[..., 1]
    thr, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coverage = float((sat > thr).mean())

    warnings = []
    if sharpness < BLUR_MIN:
        warnings.append("blur")
    if brightness < DARK_MAX:
        warnings.append("dark")
    if brightness > BRIGHT_MIN:
        warnings.append("bright")
    if coverage < COVER_MIN:
        warnings.append("small")
    if coverage > COVER_MAX:
        warnings.append("busy")

    return {"sharpness": sharpness, "brightness": brightness, "coverage": coverage}, warnings


st.set_page_config(page_title="Olive Leaf Disease Detector", page_icon="🌿")

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #4B5D16; }
    [data-testid="stSidebar"] { background-color: #E8EBD5; }
    </style>
    """,
    unsafe_allow_html=True,
)

label = st.sidebar.radio("Language / Langue / اللغة", list(LANG_LABELS.keys()))
lang = LANG_LABELS[label]
T = TEXT[lang]
rtl = lang == "ar"


def show(text, kind="markdown"):
    """Render text, right-to-left for Arabic."""
    if rtl:
        st.markdown(f'<div dir="rtl" style="text-align:right">{text}</div>', unsafe_allow_html=True)
    elif kind == "caption":
        st.caption(text)
    else:
        st.markdown(text)


show(f"# {T['title']}")
show(T["warn"], "caption")

file = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    metrics, issues = quality_checks(img)
    if issues:
        st.warning("**" + T["q_title"] + ":** " + " ".join(T["q"][k] for k in issues))
    with st.expander(T["q_title"]):
        st.caption(
            f"sharpness={metrics['sharpness']:.0f}, brightness={metrics['brightness']:.0f}, "
            f"leaf coverage={metrics['coverage']:.2f} (heuristic thresholds, not tuned)"
        )

    probs = predict(load_model(), img)
    top = max(range(3), key=lambda i: probs[i])
    key = CLASS_KEYS[top]

    show(f"### {T['pred']}: {T['names'][key]}")
    show(f"**{T['conf']}:** {probs[top]:.1f}%")

    if probs[top] < LOW_CONF:
        st.warning(T["low"])

    show(f"**{T['probs']}**")
    for i, k in enumerate(CLASS_KEYS):
        st.progress(min(max(probs[i] / 100, 0.0), 1.0), text=f"{T['names'][k]}: {probs[i]:.1f}%")

    show(f"**{T['cam_title']}**")
    st.image(gradcam_overlay(load_model(), img, top), caption=T["cam_caption"], use_container_width=True)

    desc, treat, prev = T["info"][key]
    show(f"**{T['desc']}:** {desc}")
    if treat:
        show(f"**{T['treat']}:** {treat}")
        show(f"**{T['prev']}:** {prev}")
        show(T["advice"], "caption")

    st.divider()
    show(f"**{T['fb_title']}**")
    choice = st.radio(T["fb_ask"], [T["fb_yes"], T["fb_no"]], index=None, horizontal=True)
    if choice is not None:
        if choice == T["fb_yes"]:
            true_key = key
        else:
            true_key = st.selectbox(T["fb_true"], CLASS_KEYS, format_func=lambda k: T["names"][k])
        show(T["fb_note"], "caption")
        area = st.radio(T["fb_area"], [T["fb_yes"], T["fb_no"], T["fb_unsure"]], index=None, horizontal=True)
        area_code = {T["fb_yes"]: "yes", T["fb_no"]: "no", T["fb_unsure"]: "unsure"}.get(area, "")
        consent = st.checkbox(T["fb_consent"])
        if st.button(T["fb_save"], disabled=not consent):
            save_feedback(img, true_key, key, probs[top], file.name, area_code)
            st.success(T["fb_saved"])
