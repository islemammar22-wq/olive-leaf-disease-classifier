import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "olive_multiclass_v3.pth"
CLASS_KEYS = ["healthy", "aculus", "peacock"]  # must match training order: Healthy, aculus_olearius, olive_peacock_spot
LOW_CONF = 70.0  # percent

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


st.set_page_config(page_title="Olive Leaf Disease Detector", page_icon="🌿")

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

    desc, treat, prev = T["info"][key]
    show(f"**{T['desc']}:** {desc}")
    if treat:
        show(f"**{T['treat']}:** {treat}")
        show(f"**{T['prev']}:** {prev}")
        show(T["advice"], "caption")
