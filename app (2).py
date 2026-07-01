import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")  # headless backend, obligatoire sur un serveur sans écran
import matplotlib.pyplot as plt
import base64
import os

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemicCheck - Détection d'Anémie",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CSS للواجهة ==========
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
    }
    
    .logo-container {
        text-align: center;
        padding: 20px;
        background: transparent;
        border-radius: 30px;
        margin-bottom: 10px;
    }
    
    .logo-image {
        width: 100%;
        max-width: 500px;
        height: auto;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        transition: transform 0.4s ease;
        filter: drop-shadow(0 15px 25px rgba(220,38,38,0.2));
    }
    
    .logo-image:hover {
        transform: scale(1.02);
    }
    
    .app-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(135deg, #8b0000, #dc2626, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
        letter-spacing: 2px;
    }
    
    .app-subtitle-ar {
        text-align: center;
        font-size: 28px;
        color: #b91c1c;
        margin: 5px 0;
        font-weight: 600;
    }
    
    .app-subtitle {
        text-align: center;
        font-size: 20px;
        color: #dc2626;
        margin: 5px 0;
        font-style: italic;
    }
    
    .app-tagline {
        text-align: center;
        font-size: 16px;
        color: #6c757d;
        margin-top: 5px;
        padding-bottom: 15px;
        border-bottom: 2px solid #dc2626;
        display: inline-block;
        width: auto;
    }
    
    .tagline-container {
        text-align: center;
        margin-bottom: 20px;
    }
    
    .card {
        background: white;
        border-radius: 24px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 20px 35px -10px rgba(220,38,38,0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(220,38,38,0.2);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 40px -12px rgba(220,38,38,0.25);
    }
    
    .model-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #ffcccc;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(220,38,38,0.15);
        border-color: #dc2626;
    }
    
    .model-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    
    .model-title {
        font-size: 22px;
        font-weight: 700;
        color: #dc2626;
        margin: 10px 0;
    }
    
    .model-desc {
        font-size: 14px;
        color: #6c757d;
        line-height: 1.5;
    }
    
    .model-badge {
        display: inline-block;
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        margin-top: 10px;
    }
    
    .result-anemia {
        background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border: 2px solid #dc2626;
        box-shadow: 0 10px 25px rgba(220,38,38,0.15);
    }
    
    .result-non-anemia {
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border: 2px solid #16a34a;
        box-shadow: 0 10px 25px rgba(22,163,74,0.15);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #dc2626, #ef4444, #f87171);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 14px 35px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 10px rgba(220,38,38,0.3);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #b91c1c, #dc2626, #ef4444);
        box-shadow: 0 6px 15px rgba(220,38,38,0.4);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        border-radius: 10px;
    }
    
    .disclaimer {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        font-size: 12px;
        color: #495057;
        margin-top: 30px;
        border-left: 4px solid #dc2626;
    }
    
    .stImage {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
        border-left: 4px solid #dc2626;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #ffcccc;
    }
    
    .stRadio > div {
        gap: 30px;
        justify-content: center;
    }
    
    .stRadio label {
        font-size: 16px;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #dc2626;
    }
    
    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #dc2626;
        margin: 20px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #dc2626;
        display: inline-block;
    }
    
    .section-container {
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ========== JavaScript لطلب الكاميرا ==========
st.markdown("""
<script>
    async function requestCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            console.log("Camera access granted");
            stream.getTracks().forEach(track => track.stop());
        } catch (err) {
            console.log("Camera access denied: " + err);
        }
    }
    requestCamera();
</script>
""", unsafe_allow_html=True)

# ========== عرض الشعار ==========
def get_logo_base64():
    possible_names = ["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png", "Logo.png", "logo.PNG"]
    for logo_path in possible_names:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode(), logo_path
    return None, None

logo_base64, logo_name = get_logo_base64()

if logo_base64:
    st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" class="logo-image">
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="logo-container">
        <div style="font-size: 150px; text-align: center;">🩸👁️</div>
    </div>
    """, unsafe_allow_html=True)

# عنوان التطبيق
st.markdown("""
<div class="app-title">AnemicCheck</div>
<div class="app-subtitle-ar">رؤية ميد</div>
<div class="app-subtitle">Détection d'Anémie par Intelligence Artificielle</div>
<div class="tagline-container">
    <div class="app-tagline">فحص فقر الدم غير الجراحي • Dépistage non invasif</div>
</div>
""", unsafe_allow_html=True)

# ========== عرض النماذج المستخدمة ==========
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<span class="section-title">🧠 Modèles d\'IA utilisés / النماذج المستخدمة</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

col_model1, col_model2 = st.columns(2)

with col_model1:
    st.markdown("""
    <div class="model-card">
        <div class="model-icon">🖼️🔬</div>
        <div class="model-title">U-Net</div>
        <div class="model-desc">
            <b>Segmentation / تجزئة الصور</b><br>
            Encoder: ResNet34<br>
            Extraction de la conjonctive / استخراج الملتحمة<br>
            <span class="model-badge">Segmentation model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_model2:
    st.markdown("""
    <div class="model-card">
        <div class="model-icon">🧠📊</div>
        <div class="model-title">EfficientNet-B3</div>
        <div class="model-desc">
            <b>Classification / تصنيف</b><br>
            Architecture EfficientNet<br>
            Détection d'anémie / كشف فقر الدم<br>
            <span class="model-badge">Classification model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(220,38,38,0.05); border-radius: 15px; padding: 15px; margin: 10px 0; text-align: center;">
    <p style="color: #666; margin: 0;">
        <b>📊 Pipeline de traitement / سير المعالجة:</b><br>
        Image → U-Net (Segmentation) → Extraction Conjonctive → EfficientNet-B3 (Classification) → Diagnostic
    </p>
</div>
""", unsafe_allow_html=True)

# ========== تحميل النماذج ==========
@st.cache_resource
def load_unet_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import segmentation_models_pytorch as smp
    
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    unet.load_state_dict(torch.load("Unet_model_.pth", map_location=device, weights_only=True))
    unet.to(device)
    unet.eval()
    
    return unet, device

@st.cache_resource
def load_classifier_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    
    checkpoint = torch.load("efficientnet_b3.pth", map_location=device, weights_only=True)
    
    if 'classifier.1.weight' in checkpoint:
        del checkpoint['classifier.1.weight']
    if 'classifier.1.bias' in checkpoint:
        del checkpoint['classifier.1.bias']
    
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    return model, device

# تحميل النماذج
with st.spinner("🔃 Chargement des modèles intelligents..."):
    try:
        unet_model, unet_device = load_unet_model()
    except Exception as e:
        st.error(f"❌ Erreur de chargement U-Net: {e}")
        st.stop()

    try:
        classifier_model, classifier_device = load_classifier_model()
    except Exception as e:
        st.error(f"❌ Erreur de chargement EfficientNet-B3: {e}")
        st.stop()

st.success("✅ Modèles chargés avec succès")

# ========== دالة تنظيف الماسك ==========
def clean_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, -1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    
    return clean

def extract_best_conjunctiva(img, mask):
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        if w > 0 and h > 0:
            cropped = img[y:y+h, x:x+w]
            return cropped, mask, (x, y, w, h)
    
    conjunctiva = cv2.bitwise_and(img, img, mask=mask)
    return conjunctiva, mask, None

# ========== دالة التصنيف (عكس النتيجة بشكل قسري) ==========
def predict_anemia(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()
    
    # ==================================================
    # تصحيح النتيجة: النموذج يعطي نتائج معكوسة
    # المصاب يعطيه نسبة منخفضة، غير المصاب نسبة عالية
    # لذلك نعكس النتيجة بشكل قسري
    # ==================================================
    
    # عكس النسبة
    corrected_prediction = 1 - prediction
    
    if corrected_prediction >= 0.5:
        result = "Anemic"
        confidence = corrected_prediction * 100
    else:
        result = "Non Anemic"
        confidence = (1 - corrected_prediction) * 100
    
    return result, confidence, corrected_prediction, prediction

# ========== واجهة التطبيق ==========
st.markdown("---")
st.markdown("### 📸 Méthode d'acquisition")

option = st.radio(
    "Choisissez la méthode:",
    ["📁 Télécharger une image", "📷 Prendre une photo"],
    horizontal=True
)

uploaded = None

if option == "📁 Télécharger une image":
    uploaded = st.file_uploader("Téléchargez une image de l'œil", type=["jpg", "png", "jpeg"])
else:
    uploaded = st.camera_input("📷 Prenez une photo de l'œil", disabled=False)

# ========== معالجة الصورة ==========
if uploaded is not None:
    with st.spinner("🔍 Analyse de l'image en cours..."):
        # قراءة الصورة
        img = np.array(Image.open(uploaded).convert('RGB'))
        
        # تصحيح انعكاس الصورة
        img = cv2.flip(img, 1)
        
        # تجزئة U-Net (بدون albumentations، فقط torchvision)
        transform_unet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        img_tensor = transform_unet(img).unsqueeze(0).to(unet_device)
        
        with torch.no_grad():
            raw_mask = torch.sigmoid(unet_model(img_tensor)).squeeze().cpu().numpy()
            raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))
        
        # تحسين الماسك
        cleaned_mask = clean_mask(raw_mask)
        conjunctiva, final_mask, bbox = extract_best_conjunctiva(img, cleaned_mask)
        
        # تصنيف الأنيميا (مع التصحيح)
        result, confidence, corrected_pred, raw_pred = predict_anemia(classifier_model, conjunctiva, classifier_device)
        
        # حساب النسب للمخطط (بعد التصحيح)
        anemia_percent = corrected_pred * 100
        non_anemia_percent = (1 - corrected_pred) * 100
        
        # عرض النتائج
        st.markdown("## 📊 Résultats de l'analyse")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🖼️ Image originale**")
            st.image(img, use_container_width=True)
        
        with col2:
            st.markdown("**🎭 Segmentation (U-Net)**")
            st.image(final_mask, use_container_width=True, clamp=True)
        
        with col3:
            st.markdown("**👁️ Conjonctive extraite**")
            st.image(conjunctiva, use_container_width=True)
        
        # إحصائيات
        before_area = np.sum(raw_mask > 0) / 255
        after_area = np.sum(final_mask > 0) / 255
        reduction = ((before_area - after_area) / before_area * 100) if before_area > 0 else 0
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Surface segmentée", f"{after_area:.0f} px²")
        with col_stat2:
            st.metric("Nettoyage", f"{reduction:.1f}% d'artefacts")
        
        # نتيجة التشخيص
        st.markdown("---")
        st.markdown("## 🩺 Diagnostic (EfficientNet-B3)")
        
        col_result, col_conf = st.columns(2)
        
        with col_result:
            if result == "Anemic":
                st.markdown("""
                <div class="result-anemia">
                    <h2 style="color: #dc2626; font-size: 32px;">🩸 Anémie</h2>
                    <p style="font-size: 18px;"><b>Anémie détectée</b></p>
                    <p style="font-size: 14px; color: #666;">يوجد فقر دم</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-non-anemia">
                    <h2 style="color: #16a34a; font-size: 32px;">✅ Non Anémie</h2>
                    <p style="font-size: 18px;"><b>Pas d'anémie détectée</b></p>
                    <p style="font-size: 14px; color: #666;">لا يوجد فقر دم</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_conf:
            st.metric("Niveau de confiance", f"{confidence:.1f}%")
            st.progress(int(confidence))
        
        # مخطط الأعمدة
        st.markdown("### 📈 Niveau d'anémie")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        categories = ['Non anemic', 'Anemic']
        values = [non_anemia_percent, anemia_percent]
        colors = ['#10b981', '#dc2626']
        
        bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pourcentage (%)', fontsize=12)
        ax.set_title('Probabilité d\'anémie (après correction)', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=14, 
                    color='#8b0000' if val == anemia_percent else '#166534')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
        # معلومات إضافية
        with st.expander("📈 Détails techniques"):
            st.write(f"**Diagnostic:** {result}")
            st.write(f"**Niveau de confiance:** {confidence:.2f}%")
            st.write(f"**Valeur brute (sigmoid originale):** {raw_pred:.4f}")
            st.write(f"**Valeur corrigée:** {corrected_pred:.4f}")
            st.write(f"**Règle de correction:** 1 - valeur brute (car modèle inversé)")
            st.write(f"**Anémie (probabilité corrigée):** {anemia_percent:.1f}%")
            st.write(f"**Non Anémie (probabilité corrigée):** {non_anemia_percent:.1f}%")
            st.write(f"**Appareil utilisé:** {'GPU' if classifier_device.type == 'cuda' else 'CPU'}")
            st.write(f"**Modèle de segmentation:** U-Net (ResNet34)")
            st.write(f"**Modèle de classification:** EfficientNet-B3")
        
        # تنويه طبي
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Avertissement médical / تنويه طبي</b><br>
            Ce résultat est fourni à titre indicatif et ne remplace pas un avis médical professionnel.<br>
            Veuillez consulter un médecin pour un diagnostic fiable.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: white; border-radius: 24px; margin: 20px 0;">
        <div style="font-size: 48px; margin-bottom: 20px;">📸</div>
        <h3>Bienvenue sur AnemicCheck</h3>
        <p>Sélectionnez une méthode d'acquisition ci-dessus pour commencer l'analyse</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Guide d'utilisation"):
        st.markdown("""
        **Comment fonctionne l'application ?**
        
        1. **U-Net** : Segmentation avancée pour extraire la conjonctive
        2. **EfficientNet-B3** : Classification pour détecter l'anémie
        
        **Conseils pour une analyse optimale :**
        - 📷 Utilisez une image claire et bien éclairée
        - 👁️ Assurez-vous que l'œil est bien visible
        - 🩸 Les résultats sont à titre indicatif
        
        **Technologies utilisées :**
        - PyTorch pour l'IA
        - Streamlit pour l'interface
        - OpenCV pour le traitement d'image
        """)