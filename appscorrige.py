# import os, tempfile
# from datetime import date, datetime
# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import cv2
# from tensorflow.keras.applications.vgg16 import preprocess_input

# st.set_page_config(page_title="BreastCare AI", page_icon="🩺", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap');
# .stApp{background:linear-gradient(135deg,#0b111b,#142235);color:#f3f6fb;font-family:'DM Sans',sans-serif}
# [data-testid="stSidebar"]{background:#111b2a;border-right:1px solid #29394d}
# .block-container{max-width:1450px;padding-top:1.4rem}
# h1,h2,h3{color:#f6f8fc}
# div[data-testid="stMetric"]{background:#1c2a3d;border:1px solid #30445e;padding:16px;border-radius:15px}
# .stButton>button{background:#1c3552;color:white;border:1px solid #42658b;border-radius:10px;font-weight:700}
# .stButton>button:hover{background:#244b73;color:white;border-color:#56a5ff}
# .hero{padding:38px;border-radius:22px;min-height:330px;background:linear-gradient(90deg,rgba(7,14,24,.96),rgba(7,14,24,.76),rgba(7,14,24,.2)),url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1800&q=85') center/cover;border:1px solid #34465b}
# .hero h1{font-size:42px;line-height:1.1}.hero p{max-width:620px;color:#d6e0ed;font-size:16px;line-height:1.7}
# .panel{background:#1a2738;border:1px solid #304258;border-radius:16px;padding:18px}
# .muted{color:#9eafc4}.alert{background:#3a2029;border:1px solid #8c3d4c;padding:14px;border-radius:12px;color:#ffe0e4}
# .info{background:#172f49;border:1px solid #315d86;padding:14px;border-radius:12px;color:#dceeff}
# .card{background:#1a2738;border:1px solid #304258;border-radius:16px;padding:18px;min-height:170px}
# .small{font-size:11px;letter-spacing:1.5px;color:#72c7e5;font-weight:800}
# </style>
# """, unsafe_allow_html=True)

# if "page" not in st.session_state: st.session_state.page="Accueil"
# if "records" not in st.session_state: st.session_state.records=[]
# if "last" not in st.session_state: st.session_state.last=None
# if "model_file" not in st.session_state: st.session_state.model_file=None

# @st.cache_resource
# def load_model(path):
#     return tf.keras.models.load_model(path)

# def gradcam(model, batch, class_idx):
#     base=model.layers[0]
#     conv=base.get_layer("block5_conv3")
#     conv_model=tf.keras.Model(base.input,conv.output)
#     with tf.GradientTape() as tape:
#         activations=conv_model(batch,training=False)
#         tape.watch(activations)
#         x=activations
#         for layer in model.layers[1:]:
#             x=layer(x,training=False) if isinstance(layer,tf.keras.layers.Dropout) else layer(x)
#         score=x[:,class_idx]
#     grads=tape.gradient(score,activations)
#     weights=tf.reduce_mean(grads,axis=(0,1,2))
#     heat=tf.reduce_sum(activations[0]*weights,axis=-1)
#     heat=tf.maximum(heat,0)
#     heat=heat/(tf.reduce_max(heat)+1e-8)
#     return heat.numpy()

# def prep(image):
#     im=image.convert("RGB").resize((224,224))
#     arr=np.asarray(im,dtype=np.float32)
#     batch=preprocess_input(arr.copy())[None,...]
#     return batch,arr.astype(np.uint8)

# def overlay_gradcam(original,heat,alpha=.4):
#     h,w=original.shape[:2]
#     heat=cv2.resize(heat,(w,h))
#     heat8=np.uint8(np.clip(heat*255,0,255))
#     color=cv2.cvtColor(cv2.applyColorMap(heat8,cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
#     overlay=cv2.addWeighted(original,.6,color,.4,0)
#     return heat,overlay

# with st.sidebar:
#     st.markdown("## 🩺 BreastCare AI")
#     st.caption("Plateforme expérimentale d'imagerie mammaire")
#     up_model=st.file_uploader("Importer modele_vgg16.keras",type=["keras"])
#     if up_model is not None and st.session_state.get("model_name")!=up_model.name:
#         tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".keras")
#         tmp.write(up_model.getvalue()); tmp.close()
#         st.session_state.model_file=tmp.name
#         st.session_state.model_name=up_model.name
#     local_path=st.text_input("Ou chemin du modèle", "modele_vgg16.keras")
#     path=st.session_state.model_file or (local_path if os.path.exists(local_path) else None)
#     model=None
#     if path:
#         try: model=load_model(path); st.success("Modèle chargé")
#         except Exception as e: st.error(f"Erreur de chargement : {e}")
#     else: st.warning("Chargez le fichier .keras pour activer l'analyse.")
#     st.markdown("---")
#     st.markdown("### Navigation")
#     for p,lab in [("Accueil","⌂ Accueil"),("Tableau de bord","▦ Tableau de bord"),("Analyse IA","✣ Analyse échographie"),("Patients","♙ Profils patients"),("Rapports","▤ Rapports"),("À propos","ⓘ À propos")]:
#         if st.button(lab,use_container_width=True,key="nav_"+p):
#             st.session_state.page=p
#     st.markdown("---")
#     st.markdown("### Classes du modèle")
#     names_text=st.text_area("Noms dans l'ordre exact d'entraînement", "Classe 0\nClasse 1\nClasse 2")
#     class_names=[s.strip() for s in names_text.splitlines() if s.strip()]
#     if len(class_names)!=3:
#         st.warning("Saisir exactement trois classes.")
#         class_names=["Classe 0","Classe 1","Classe 2"]
#     st.caption("Prototype académique — non validé pour le diagnostic.")

# page=st.session_state.page
# st.markdown("<div class='small'>BREAST IMAGING • DEEP LEARNING • CLINICAL WORKFLOW</div>",unsafe_allow_html=True)

# if page=="Accueil":
#     st.markdown("""
#     <div class="hero">
#       <div class="small">INTELLIGENCE ARTIFICIELLE • IMAGERIE MAMMAIRE</div>
#       <h1>Une lecture augmentée<br>de l'imagerie du sein.</h1>
#       <p>BreastCare AI associe classification d'images par VGG16 et visualisation Grad-CAM
#       pour explorer les régions qui influencent les prédictions du modèle.</p>
#       <span style="background:#1d4774;padding:8px 13px;border-radius:20px">VGG16 · 3 classes · Grad-CAM</span>
#     </div>""",unsafe_allow_html=True)
#     st.write("")
#     a,b,c=st.columns(3)
#     with a: st.markdown("<div class='card'><div class='small'>ESPACE PROFESSIONNEL</div><h3>👩‍⚕️ Médecin / Radiologue</h3><p class='muted'>Importer une image, examiner les scores et saisir son avis professionnel.</p></div>",unsafe_allow_html=True)
#     with b: st.markdown("<div class='card'><div class='small'>DOSSIER PATIENT</div><h3>🧑‍🦰 Profil patient</h3><p class='muted'>Créer un profil anonymisé et retrouver les analyses enregistrées.</p></div>",unsafe_allow_html=True)
#     with c: st.markdown("<div class='card'><div class='small'>EXPLICABILITÉ IA</div><h3>🔥 Grad-CAM</h3><p class='muted'>Visualiser les régions contribuant le plus au score de la classe choisie.</p></div>",unsafe_allow_html=True)
#     st.write("")
#     st.markdown("<div class='info'>Les résultats sont expérimentaux. Ils ne constituent pas un diagnostic et doivent être revus par un professionnel qualifié.</div>",unsafe_allow_html=True)

# elif page=="Analyse IA":
#     st.title("✣ Analyse intelligente d'une échographie")
#     st.caption("Dossier patient · importation · classification · Grad-CAM · avis professionnel")
#     if model is None: st.markdown("<div class='alert'>Chargez d'abord votre fichier modele_vgg16.keras dans la barre latérale.</div>",unsafe_allow_html=True)
#     left,right=st.columns(2)
#     with left:
#         st.markdown("<div class='panel'>",unsafe_allow_html=True)
#         st.subheader("1. Informations patient")
#         pid=st.text_input("Identifiant patient *",placeholder="BC-2026-001")
#         pname=st.text_input("Nom ou code anonymisé",placeholder="Patiente A.")
#         age=st.number_input("Âge",0,120,35)
#         exam_date=st.date_input("Date de l'examen",date.today())
#         doctor=st.text_input("Médecin / opérateur")
#         st.markdown("</div>",unsafe_allow_html=True)
#     with right:
#         st.markdown("<div class='panel'>",unsafe_allow_html=True)
#         st.subheader("2. Image échographique")
#         up=st.file_uploader("Importer PNG, JPG ou JPEG",type=["png","jpg","jpeg"])
#         if up:
#             pil=Image.open(up).convert("RGB")
#             st.image(pil,caption="Image importée",use_container_width=True)
#         st.markdown("</div>",unsafe_allow_html=True)
#     if st.button("Lancer l'analyse expérimentale",type="primary",use_container_width=True,disabled=(model is None or up is None)):
#         if not pid.strip(): st.error("Identifiant patient obligatoire.")
#         else:
#             try:
#                 batch,original=prep(pil)
#                 probs=model.predict(batch,verbose=0)[0]
#                 idx=int(np.argmax(probs))
#                 heat=gradcam(model,batch,idx)
#                 heat,overlay=overlay_gradcam(original,heat)
#                 st.session_state.last={"ID":pid.strip(),"Patient":pname or "Non renseigné","Âge":int(age),
#                     "Date examen":str(exam_date),"Médecin":doctor or "Non renseigné",
#                     "Classe modèle":class_names[idx],"Score modèle (%)":round(float(probs[idx])*100,2),
#                     "probabilities":probs.tolist(),"image":original,"heatmap":heat,"overlay":overlay,
#                     "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M")}
#                 st.success("Analyse terminée.")
#             except Exception as e: st.error(f"Erreur pendant l'analyse : {e}")
#     r=st.session_state.last
#     if r:
#         st.markdown("---"); st.subheader("3. Résultats du modèle")
#         x,y,z=st.columns(3)
#         x.metric("Classe prédite",r["Classe modèle"]); y.metric("Score du modèle",f'{r["Score modèle (%)"]:.2f}%'); z.metric("Identifiant",r["ID"])
#         st.markdown("<div class='alert'><b>Important :</b> le score est une sortie du modèle, pas une probabilité clinique validée ni un diagnostic.</div>",unsafe_allow_html=True)
#         st.subheader("Probabilités par classe")
#         st.bar_chart(pd.DataFrame({"Classe":class_names,"Score (%)":np.array(r["probabilities"])*100}).set_index("Classe"))
#         st.subheader("4. Explicabilité Grad-CAM")
#         i1,i2,i3=st.columns(3)
#         i1.image(r["image"],caption="Image originale",use_container_width=True)
#         i2.image(r["heatmap"],caption="Carte thermique",use_container_width=True,clamp=True)
#         i3.image(r["overlay"],caption="Superposition Grad-CAM",use_container_width=True)
#         st.subheader("5. Avis du professionnel")
#         status=st.selectbox("Statut de revue",["En attente de revue médicale","À corréler avec l'examen clinique","Examen complémentaire à discuter","Avis documenté par le médecin"])
#         note=st.text_area("Commentaire du médecin",placeholder="Saisie de l'observation clinique par le professionnel.")
#         if st.button("Enregistrer le dossier"):
#             rec={k:v for k,v in r.items() if k not in ["probabilities","image","heatmap","overlay"]}
#             rec["Avis professionnel"]=status; rec["Commentaire"]=note
#             st.session_state.records=[p for p in st.session_state.records if p["ID"]!=rec["ID"]]
#             st.session_state.records.append(rec)
#             st.success("Dossier enregistré dans la session.")

# elif page=="Tableau de bord":
#     st.title("▦ Tableau de bord hospitalier")
#     df=pd.DataFrame(st.session_state.records)
#     total=len(df); pending=int(df["Avis professionnel"].eq("En attente de revue médicale").sum()) if total else 0
#     reviewed=total-pending; avg=df["Score modèle (%)"].mean() if total else 0
#     a,b,c,d=st.columns(4)
#     a.metric("Dossiers enregistrés",total); b.metric("Avis documentés",reviewed); c.metric("En attente",pending); d.metric("Score moyen IA",f"{avg:.1f}%" if total else "—")
#     st.write("")
#     if total:
#         l,r=st.columns(2)
#         with l: st.subheader("Classes prédites"); st.bar_chart(df["Classe modèle"].value_counts())
#         with r: st.subheader("Statut des dossiers"); st.bar_chart(df["Avis professionnel"].value_counts())
#         st.subheader("Derniers dossiers"); st.dataframe(df,use_container_width=True,hide_index=True)
#         if pending: st.warning(f"{pending} dossier(s) attendent une revue professionnelle.")
#     else: st.info("Aucun dossier enregistré. Réalisez une analyse puis enregistrez le compte rendu.")

# elif page=="Patients":
#     st.title("♙ Profils patients")
#     with st.form("new_patient"):
#         a,b=st.columns(2)
#         with a: pid=st.text_input("Identifiant unique *"); name=st.text_input("Nom/code anonymisé"); age=st.number_input("Âge",0,120,30)
#         with b: sex=st.selectbox("Sexe",["Femme","Homme","Autre / non renseigné"]); created=st.date_input("Date de création",date.today())
#         submit=st.form_submit_button("Créer le profil")
#         if submit:
#             if not pid.strip(): st.error("Identifiant obligatoire.")
#             elif any(x["ID"]==pid.strip() for x in st.session_state.records): st.error("Identifiant déjà présent.")
#             else:
#                 st.session_state.records.append({"ID":pid.strip(),"Patient":name or "Non renseigné","Âge":int(age),"Sexe":sex,"Date création":str(created),"Classe modèle":"Aucune analyse","Score modèle (%)":np.nan,"Avis professionnel":"Profil créé — analyse en attente","Commentaire":"","Horodatage":datetime.now().strftime("%Y-%m-%d %H:%M")})
#                 st.success("Profil créé.")
#     df=pd.DataFrame(st.session_state.records)
#     if len(df):
#         st.subheader("Registre des patients"); st.dataframe(df,use_container_width=True,hide_index=True)
#         chosen=st.selectbox("Ouvrir un dossier",df["ID"].tolist())
#         row=df[df["ID"]==chosen].iloc[-1]
#         st.markdown("<div class='panel'>",unsafe_allow_html=True)
#         st.subheader(f"Dossier {chosen}")
#         st.write("Patient :",row.get("Patient","—"),"| Âge :",row.get("Âge","—"))
#         st.write("Classe modèle :",row.get("Classe modèle","—"))
#         st.write("Avis professionnel :",row.get("Avis professionnel","—"))
#         st.write("Commentaire :",row.get("Commentaire","—"))
#         st.markdown("</div>",unsafe_allow_html=True)
#     else: st.info("Aucun profil. Créez un profil ou enregistrez une analyse.")

# elif page=="Rapports":
#     st.title("▤ Rapports et export")
#     df=pd.DataFrame(st.session_state.records)
#     if len(df):
#         st.dataframe(df,use_container_width=True,hide_index=True)
#         st.download_button("⬇ Télécharger les dossiers CSV",df.to_csv(index=False).encode("utf-8-sig"),"breastcare_patients.csv","text/csv",use_container_width=True)
#     else: st.info("Aucun dossier à exporter.")

# elif page=="À propos":
#     st.title("ⓘ À propos de BreastCare AI")
#     st.markdown("""
#     <div class='panel'>
#     <h3>Technologies</h3><p>Python · TensorFlow/Keras · VGG16 · OpenCV · Streamlit · Grad-CAM</p>
#     <h3>Objectif</h3><p>Prototype académique de classification d'images échographiques et d'explicabilité visuelle.</p>
#     <h3>Limites et sécurité</h3>
#     <ul>
#     <li>Les scores du modèle ne sont pas des probabilités cliniques validées.</li>
#     <li>Grad-CAM est une explication approximative, pas une preuve de localisation d'une lésion.</li>
#     <li>Ne pas utiliser pour diagnostiquer, trier ou traiter une patiente.</li>
#     <li>Validation clinique, réglementaire, cybersécurité et protection des données indispensables avant tout usage réel.</li>
#     <li>Les dossiers restent dans la session Streamlit et ne sont pas une base de données hospitalière sécurisée.</li>
#     </ul></div>""",unsafe_allow_html=True)

# st.markdown("---")
# st.markdown("<p style='text-align:center;color:#8192a8;font-size:12px'>BREASTCARE AI · Prototype académique · VGG16 + Grad-CAM</p>",unsafe_allow_html=True)
# Fin 
# Fin conserver

import os
import base64
import html as html_lib
import hashlib
import io
import tempfile
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# ============================================================
# 1. CONFIGURATION GENERALE
# ============================================================

st.set_page_config(
    page_title="BreastCare AI | Innovation médicale",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent
HERO_PATH = BASE_DIR / "hero.jpg"

if tuple(
    int(x) for x in st.__version__.split(".")[:2]
) >= (1, 50):
    FIT = {"width": "stretch"}
else:
    FIT = {"use_container_width": True}


# ============================================================
# 2. FONCTIONS HTML ET IMAGES
# ============================================================

def render(block):
    st.markdown(
        " ".join(
            line.strip()
            for line in block.strip().splitlines()
            if line.strip()
        ),
        unsafe_allow_html=True
    )


@st.cache_data
def hero_data_uri(path):
    path = Path(path)

    if not path.exists():
        return ""

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


@st.cache_data
def gallery_data_uri(path):
    path = Path(path)

    if not path.exists():
        return ""

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ------------------------------------------------------------
# Graphique en anneau des scores (Vega-Lite intégré à Streamlit)
# ------------------------------------------------------------

SCORE_COLORS = ["#082B49", "#1769AA", "#D5AE61"]


def score_donut_spec(names, percents):
    """Graphique en anneau (Vega-Lite, inclus dans Streamlit)."""

    values = [
        {
            "Classe": n,
            "Score": round(float(p), 2),
            "Label": f"{p:.1f} %",
            "Ordre": i,
            "R": 158
        }
        for i, (n, p) in enumerate(zip(names, percents))
    ]

    # Les petites parts sont décalées pour que leurs étiquettes ne se chevauchent pas
    small = [v for v in values if v["Score"] < 8]
    for k, v in enumerate(small):
        v["R"] = 158 if k % 2 == 0 else 192

    best = max(values, key=lambda v: v["Score"])

    stack_theta = {
        "field": "Score",
        "type": "quantitative",
        "stack": True
    }

    order = {"field": "Ordre", "type": "quantitative"}

    return {
        "width": "container",
        "height": 410,
        "background": "transparent",
        "padding": {"left": 10, "right": 10, "top": 10, "bottom": 10},
        "data": {"values": values},
        "view": {"stroke": None},
        "config": {
            "font": "DM Sans, sans-serif"
        },
        "layer": [
            {
                "mark": {
                    "type": "arc",
                    "innerRadius": 78,
                    "outerRadius": 130,
                    "stroke": "#FFFFFF",
                    "strokeWidth": 3,
                    "cornerRadius": 4
                },
                "encoding": {
                    "theta": stack_theta,
                    "order": order,
                    "color": {
                        "field": "Classe",
                        "type": "nominal",
                        "scale": {
                            "domain": list(names),
                            "range": SCORE_COLORS
                        },
                        "legend": None
                    },
                    "tooltip": [
                        {"field": "Classe", "type": "nominal",
                         "title": "Classe"},
                        {"field": "Score", "type": "quantitative",
                         "title": "Score (%)", "format": ".2f"}
                    ]
                }
            },
            {
                "mark": {
                    "type": "text",
                    "fontSize": 15,
                    "fontWeight": "bold",
                    "color": "#103B60"
                },
                "encoding": {
                    "theta": stack_theta,
                    "order": order,
                    "detail": {"field": "Classe", "type": "nominal"},
                    "radius": {"field": "R", "type": "quantitative",
                               "scale": None},
                    "text": {"field": "Label", "type": "nominal"}
                }
            },
            {
                "data": {"values": [{"t": best["Label"]}]},
                "mark": {
                    "type": "text",
                    "fontSize": 30,
                    "fontWeight": "bold",
                    "color": "#082B49",
                    "dy": -8
                },
                "encoding": {"text": {"field": "t", "type": "nominal"}}
            },
            {
                "data": {"values": [{"t": best["Classe"]}]},
                "mark": {
                    "type": "text",
                    "fontSize": 12,
                    "color": "#61758B",
                    "dy": 16
                },
                "encoding": {"text": {"field": "t", "type": "nominal"}}
            }
        ]
    }


def show_vega(spec):
    try:
        st.vega_lite_chart(spec=spec, **FIT)
    except TypeError:
        st.vega_lite_chart(spec=spec, use_container_width=True)


# ============================================================
# 2b. PROFIL PROFESSIONNEL (données, QR code, CV recruteur)
# ============================================================

PROFILE_PHOTO = BASE_DIR / "profil_ibrahima.jpg"

PROFILE = {
    "name": "Ibrahima DIENG",
    "title": "Junior Data Scientist & Analyste de Données",
    "subtitle": "Statistique et Intelligence Artificielle",
    "phone": "+221 77 282 92 34",
    "phone_raw": "+221772829234",
    "email": "ibrahimadieng997@gmail.com",
    "github": "github.com/ibrahimadieng997-boop",
    "github_url": "https://github.com/ibrahimadieng997-boop",
    "location": "Thiès, Sénégal",
    "summary": (
        "Titulaire d’une licence en Statistique et Informatique "
        "Décisionnelle (UADB), je poursuis actuellement une formation "
        "en Data Science et Intelligence Artificielle au Dakar Institute "
        "of Technology (niveau équivalent Master 1). Je mobilise mes "
        "compétences en analyse statistique, Python, R, SQL, Machine "
        "Learning et Deep Learning pour transformer les données en "
        "informations exploitables, développer des modèles prédictifs "
        "et concevoir des tableaux de bord interactifs avec Power BI, "
        "afin de soutenir la prise de décision."
    ),
    "projects": [
        ("Statistique univariée et bivariée", [
            "Analyse exploratoire et traitement de données en santé "
            "mentale : étude du score de dépendance (SDS), qui mesure "
            "la gravité de la dépendance à une drogue ou à une "
            "substance psychoactive.",
            "Analyse univariée (distribution et dispersion des variables "
            "quantitatives) et analyse bivariée (corrélations entre "
            "variables, tests statistiques).",
        ]),
        ("Statistique inférentielle", [
            "Prédiction des émissions de CO2 d’un véhicule à partir des "
            "caractéristiques du moteur et de la consommation, par "
            "régression linéaire simple et multiple.",
        ]),
        ("Machine Learning", [
            "Marketing bancaire : prédiction de la souscription d’un "
            "dépôt à terme par les clients.",
            "Segmentation de patients : identification des patients "
            "seniors fortement hospitalisés, des patients obèses "
            "fumeurs et des patients adultes en suivi.",
            "Prédiction de l’état de véhicules (« venant » ou "
            "d’occasion) du site CoinAfrique à partir du prix, de "
            "l’année, de la transmission et du lieu de stationnement.",
        ]),
        ("Deep Learning", [
            "Prédiction de la progression du diabète par perceptron "
            "simple et multicouche : jeu de données Diabetes "
            "(scikit-learn, 442 patients, 10 variables).",
            "Prétraitement des données : séparation train/test et "
            "normalisation (StandardScaler).",
            "Implémentation d’un perceptron simple puis d’un perceptron "
            "multicouche (MLP) en Python, avec descente de gradient et "
            "rétropropagation.",
        ]),
    ],
    "skills": [
        ("LANGAGES & ENVIRONNEMENTS", [
            "Python", "R", "SQL", "HTML/CSS",
            "Jupyter Notebook", "Google Colab"]),
        ("MACHINE LEARNING & DEEP LEARNING", [
            "Statistique", "Apprentissage supervisé",
            "Apprentissage non supervisé", "Apprentissage profond",
            "scikit-learn", "NumPy", "Pandas", "TensorFlow", "Keras"]),
        ("ANALYSE & VISUALISATION", [
            "Power BI", "Tableau Desktop", "Excel", "Streamlit",
            "Seaborn", "Matplotlib"]),
        ("BASES DE DONNÉES & MODÉLISATION", [
            "PostgreSQL", "PowerAMC"]),
        ("COLLECTE & DATA ENGINEERING", [
            "Web scraping", "BeautifulSoup", "Selenium", "BrowserFlow",
            "Web Scraper", "Apify", "Stockage Cloud (en cours)"]),
        ("BUREAUTIQUE", ["Word", "PowerPoint"]),
    ],
    "education": [
        ("Licence 3 en Statistique et Informatique Décisionnelle "
         "(SID) – Diplômé", "Université Alioune Diop de Bambey "
         "(UADB), Bambey", "2022 – 2023", False),
        ("Licence 2 en Statistique et Informatique Décisionnelle (SID)",
         "Université Alioune Diop de Bambey (UADB), Bambey",
         "2021 – 2022", False),
        ("Licence 1 en Mathématiques, Physique, Chimie et Informatique "
         "(MPCI)", "Mathématiques, Physique, Chimie, Informatique",
         "2020 – 2021", False),
        ("Baccalauréat S2 – Mention Assez Bien",
         "Lycée de Rao, Saint-Louis, Sénégal", "2019 – 2020", False),
    ],
    "certs": [
        ("Certification en Data Science et IA – En cours",
         "Dakar Institute of Technology (École de l’Intelligence "
         "Artificielle), Dakar", "2025 – 2026", True),
        ("Certificat de Data Analyst",
         "Xarala Académie, Cité Damel", "2025 – 2026", False),
        ("Attestation de travail",
         "Centre de Santé Mentale Dalal Xel, Thiès", "2022 – 2023",
         False),
    ],
    "community_org": "Mister Bane Center",
    "community_role": "Membre actif – Actions humanitaires (en cours)",
    "community": [
        "Aide matérielle : achat et distribution de chaussures, de "
        "vêtements et de produits de première nécessité aux talibés "
        "(enfants en situation de vulnérabilité).",
        "Accompagnement social et moral : écoute, conseil et "
        "orientation des enfants.",
        "Journées solidaires : organisation d’actions pour améliorer "
        "leur quotidien (repas, hygiène, sensibilisation).",
    ],
    "languages": ["Wolof", "Français", "Anglais"],
    "interests": ["Musique", "Cinéma", "Sport", "Lecture", "Jeux"],
}


def _e(value):
    return html_lib.escape(str(value))


def make_qr_uri(text, dark="#082B49", scale=8):
    """QR code en SVG (data URI). Nécessite le paquet 'segno'."""
    try:
        import segno
    except ImportError:
        return ""

    buffer = io.BytesIO()
    segno.make(text, error="m").save(
        buffer,
        kind="svg",
        scale=scale,
        border=2,
        dark=dark,
        light="#FFFFFF",
        xmldecl=False
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def build_vcard():
    """Carte de contact vCard 3.0 (lue directement par les téléphones)."""
    p = PROFILE
    return "\r\n".join([
        "BEGIN:VCARD",
        "VERSION:3.0",
        "N:DIENG;Ibrahima;;;",
        "FN:Ibrahima DIENG",
        "TITLE:Data Scientist Junior",
        f"TEL:{p['phone_raw']}",
        f"EMAIL:{p['email']}",
        f"URL:{p['github_url']}",
        "END:VCARD",
        ""
    ])


def build_cv_html(photo_uri, qr_uri):
    """CV recruteur autonome (HTML), imprimable en PDF (A4)."""
    p = PROFILE

    photo = (
        f'<img src="{photo_uri}" alt="Photo">'
        if photo_uri else '<div class="ini">ID</div>'
    )

    qr = (
        f'<div class="qr"><img src="{qr_uri}" alt="QR code">'
        f'<span>Scannez pour ouvrir le profil</span></div>'
        if qr_uri else ""
    )

    skills = "".join(
        f'<div class="sg"><div class="st">{_e(t)}</div>'
        + "".join(f'<span class="tag">{_e(i)}</span>' for i in items)
        + "</div>"
        for t, items in p["skills"]
    )

    projects = "".join(
        f'<div class="pj"><b>{_e(t)}</b><ul>'
        + "".join(f"<li>{_e(i)}</li>" for i in items)
        + "</ul></div>"
        for t, items in p["projects"]
    )

    def timeline(rows):
        return "".join(
            f'<div class="tl{" now" if now else ""}">'
            f'<div class="tt">{_e(t)}</div>'
            f'<div class="ts">{_e(s)} · <b>{_e(d)}</b></div></div>'
            for t, s, d, now in rows
        )

    community = "".join(f"<li>{_e(i)}</li>" for i in p["community"])

    css = """
    *{box-sizing:border-box}
    body{margin:0;background:#E9F1F9;font-family:'DM Sans','Segoe UI',Arial,sans-serif;color:#103B60}
    .page{width:210mm;min-height:297mm;margin:12px auto;background:#fff;display:flex;box-shadow:0 10px 40px rgba(8,43,73,.25)}
    .side{width:72mm;background:linear-gradient(180deg,#061F38,#0A3A63 70%,#1769AA);color:#fff;padding:26px 20px}
    .main{flex:1;padding:30px 30px 26px}
    .ph{position:relative;width:150px;height:150px;margin:0 auto 18px}
    .ph:before{content:"";position:absolute;inset:0;border-radius:50%;background:conic-gradient(#1769AA,#29B6F6,#FFD23F,#F5B700,#1769AA)}
    .ph img,.ini{position:absolute;inset:7px;width:136px;height:136px;border-radius:50%;object-fit:cover;border:4px solid #fff;background:#0A3A63;display:flex;align-items:center;justify-content:center;font-size:42px;font-weight:800}
    .side h3{font-size:10.5px;letter-spacing:2px;color:#FFD23F;margin:20px 0 8px;border-bottom:1px solid rgba(255,255,255,.25);padding-bottom:5px}
    .side p{font-size:11.5px;margin:5px 0;word-break:break-all;line-height:1.45}
    .tag{display:inline-block;font-size:10px;font-weight:700;border-radius:999px;padding:3px 9px;margin:0 4px 5px 0;background:#EAF4FC;color:#103B60;border:1px solid #CFE2F3}
    .side .tag{background:rgba(255,255,255,.12);color:#fff;border-color:rgba(255,255,255,.28)}
    .qr{margin-top:20px;text-align:center;background:#fff;border-radius:14px;padding:12px}
    .qr img{width:100%;max-width:150px}
    .qr span{display:block;font-size:9.5px;color:#61758B;margin-top:4px}
    h1{margin:0;font-size:34px;letter-spacing:-1px;color:#082B49;line-height:1.05}
    h1 span{color:#D5A00A}
    .role{font-size:14.5px;font-weight:700;color:#1769AA;margin:7px 0 0}
    .sub{font-size:11px;letter-spacing:1.8px;color:#B18A40;font-weight:800;margin-top:4px}
    h2{font-size:12px;letter-spacing:2px;color:#082B49;margin:20px 0 9px;display:flex;align-items:center;gap:10px}
    h2:after{content:"";flex:1;height:2px;background:linear-gradient(90deg,#D5AE61,transparent)}
    .sum{font-size:12px;line-height:1.65;color:#2C4A66;margin:0}
    .pj{border-left:3px solid #1769AA;background:#F1F7FD;padding:7px 11px;margin-bottom:7px;border-radius:0 8px 8px 0}
    .pj b{font-size:12px;color:#082B49}
    .pj ul{margin:3px 0 0;padding-left:15px;font-size:10.8px;line-height:1.5;color:#34506B}
    .tl{margin-bottom:8px;padding-left:12px;border-left:2px solid #CFE2F3}
    .tl.now{border-left-color:#F5B700}
    .tt{font-size:12px;font-weight:800;color:#082B49}
    .ts{font-size:10.8px;color:#61758B}
    .ts b{color:#1769AA}
    ul.cm{margin:0;padding-left:16px;font-size:10.8px;line-height:1.55;color:#34506B}
    .bar{position:fixed;top:12px;right:16px}
    .bar button{background:#082B49;color:#fff;border:1px solid #D5AE61;border-radius:10px;padding:10px 18px;font-weight:700;cursor:pointer}
    @media print{body{background:#fff}.page{margin:0;box-shadow:none}.bar{display:none}
      *{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
    @page{size:A4;margin:0}
    """

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>CV – {_e(p['name'])}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{css}</style></head>
<body>
<div class="bar"><button onclick="window.print()">Imprimer / Enregistrer en PDF</button></div>
<div class="page">
  <aside class="side">
    <div class="ph">{photo}</div>
    <h3>CONTACT</h3>
    <p>📞 {_e(p['phone'])}</p>
    <p>✉️ {_e(p['email'])}</p>
    <p>🔗 {_e(p['github'])}</p>
    <p>📍 {_e(p['location'])}</p>
    <h3>COMPÉTENCES</h3>{skills}
    <h3>LANGUES</h3>
    <p>{_e(' · '.join(p['languages']))}</p>
    <h3>CENTRES D’INTÉRÊT</h3>
    <p>{_e(' · '.join(p['interests']))}</p>
    {qr}
  </aside>
  <main class="main">
    <h1>IBRAHIMA <span>DIENG</span></h1>
    <div class="role">{_e(p['title'])}</div>
    <div class="sub">{_e(p['subtitle'].upper())}</div>
    <h2>RÉSUMÉ PROFESSIONNEL</h2>
    <p class="sum">{_e(p['summary'])}</p>
    <h2>PROJETS ACADÉMIQUES</h2>{projects}
    <h2>FORMATION</h2>{timeline(p['education'])}
    <h2>CERTIFICATIONS &amp; ATTESTATIONS</h2>{timeline(p['certs'])}
    <h2>ENGAGEMENT COMMUNAUTAIRE</h2>
    <div class="tt">{_e(p['community_org'])}</div>
    <div class="ts" style="margin-bottom:5px">{_e(p['community_role'])}</div>
    <ul class="cm">{community}</ul>
  </main>
</div>
</body></html>"""


PROFILE_CSS = """
<style>
.pf-hero{display:flex;align-items:center;gap:40px;margin-top:18px;padding:40px 46px;border-radius:26px;position:relative;overflow:hidden;
    background:linear-gradient(120deg,#061F38,#0A3A63 55%,#1769AA);border:1px solid rgba(213,174,97,.55);box-shadow:0 18px 45px rgba(10,48,80,.22)}
.pf-hero:after{content:"";position:absolute;right:-90px;top:-90px;width:340px;height:340px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,255,255,.14),transparent 65%)}
.pf-photo{position:relative;flex:0 0 auto;width:208px;height:208px;z-index:1}
.pf-ring{position:absolute;inset:0;border-radius:50%;
    background:conic-gradient(#1769AA,#29B6F6,#FFD23F,#F5B700,#1769AA);
    animation:pf-spin 7s linear infinite;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.pf-photo img,.pf-initials{position:absolute;inset:9px;width:calc(100% - 18px);height:calc(100% - 18px);border-radius:50%;
    object-fit:cover;border:5px solid #fff;background:#0A3A63;box-sizing:border-box}
.pf-initials{display:flex;align-items:center;justify-content:center;color:#fff;font-size:56px;font-weight:800}
@keyframes pf-spin{to{transform:rotate(360deg)}}
.pf-id{position:relative;z-index:1}
.pf-kicker{color:#D5AE61;font-size:12px;font-weight:800;letter-spacing:2.4px}
.pf-name{color:#fff;font-size:54px;font-weight:800;letter-spacing:-1.8px;line-height:1.02;margin-top:10px}
.pf-name span{color:#FFD23F}
.pf-title{color:#D2E6F5;font-size:19px;font-weight:600;margin-top:12px}
.pf-badge{display:inline-block;margin-top:14px;background:rgba(255,210,63,.14);border:1px solid rgba(255,210,63,.55);color:#FFE27A;
    border-radius:999px;padding:6px 14px;font-size:12px;font-weight:800;letter-spacing:.8px}
.pf-chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}
.pf-chip{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.26);color:#fff !important;border-radius:999px;
    padding:8px 15px;font-size:13px;font-weight:600;text-decoration:none !important}
a.pf-chip:hover{background:#1769AA}
.pf-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0}
.pf-stat{background:#fff;border:1px solid var(--border);border-top:4px solid var(--blue);border-radius:16px;padding:16px 18px;box-shadow:0 7px 23px rgba(25,75,120,.06)}
.pf-stat:nth-child(even){border-top-color:#F5B700}
.pf-stat b{display:block;font-size:28px;color:var(--navy);font-weight:800;line-height:1.1}
.pf-stat span{font-size:12px;color:var(--muted);font-weight:700;letter-spacing:.4px}
.pf-card{background:#fff;border:1px solid var(--border);border-radius:18px;padding:24px 26px;margin-bottom:20px;box-shadow:0 7px 23px rgba(25,75,120,.06)}
.pf-h{display:flex;align-items:center;gap:10px;color:var(--navy);font-size:17px;font-weight:800;margin-bottom:15px}
.pf-h:after{content:"";flex:1;height:2px;background:linear-gradient(90deg,#D5AE61,transparent)}
.pf-ico{width:34px;height:34px;display:flex;align-items:center;justify-content:center;border-radius:10px;
    background:linear-gradient(135deg,#E7F3FD,#D6EAFB);border:1px solid #C9E0F3;font-size:17px}
.pf-sum{color:#2C4A66;line-height:1.8;font-size:15px}
.pf-proj{border-left:4px solid var(--blue);background:var(--light);border-radius:0 12px 12px 0;padding:13px 17px;margin-bottom:12px}
.pf-proj b{color:var(--navy);display:block;margin-bottom:5px}
.pf-proj ul,.pf-list{margin:0;padding-left:19px;color:#34506B;font-size:13.5px;line-height:1.7}
.pf-tl{border-left:2px solid #CFE2F3;margin-left:6px;padding-left:22px}
.pf-tl-i{position:relative;margin-bottom:17px}
.pf-tl-i:before{content:"";position:absolute;left:-31px;top:4px;width:12px;height:12px;border-radius:50%;background:#fff;border:3px solid var(--blue)}
.pf-tl-i.now:before{border-color:#F5B700;background:#FFD23F}
.pf-tl-t{font-weight:800;color:var(--navy);font-size:14.5px}
.pf-tl-s{color:var(--muted);font-size:12.5px;margin-top:2px}
.pf-tl-d{display:inline-block;background:var(--light);color:var(--blue);font-size:11.5px;font-weight:800;border-radius:999px;padding:2px 11px;margin-top:5px}
.pf-sk-t{font-size:11px;font-weight:800;letter-spacing:1.6px;color:#B18A40;margin:4px 0 8px}
.pf-tag{display:inline-block;background:var(--light);border:1px solid #CFE2F3;color:var(--dark);border-radius:999px;
    padding:5px 12px;font-size:12.5px;font-weight:700;margin:0 6px 7px 0}
.pf-org{font-weight:800;color:var(--navy);font-size:15px}
.pf-role{color:#B18A40;font-weight:800;font-size:12px;letter-spacing:.8px;margin:2px 0 9px}
.pf-scan{border-radius:22px;padding:30px 36px;margin:6px 0 20px;color:#fff;
    background:linear-gradient(110deg,#061F38,#0A3A63 60%,#0D4570);border:1px solid rgba(213,174,97,.5);box-shadow:0 18px 45px rgba(10,48,80,.15)}
.pf-scan .pf-kicker{margin-bottom:8px}
.pf-scan-t{font-size:30px;font-weight:800;letter-spacing:-.8px}
.pf-scan p{color:#D2E6F5;line-height:1.7;margin:8px 0 0;max-width:760px}
.pf-qr{text-align:center;background:#fff;border:1px solid var(--border);border-radius:20px;padding:24px 18px;box-shadow:0 7px 23px rgba(25,75,120,.06)}
.pf-qr img{width:240px;max-width:100%;padding:10px;border:4px solid transparent;border-radius:20px;
    background:linear-gradient(#fff,#fff) padding-box,linear-gradient(135deg,#1769AA,#FFD23F) border-box}
.pf-qr b{display:block;color:var(--navy);font-size:16px;margin-top:14px}
.pf-qr span{color:var(--muted);font-size:12.5px;line-height:1.5}
@media (max-width:900px){
 .pf-hero{flex-direction:column;text-align:center;padding:32px 22px}
 .pf-chips{justify-content:center}
 .pf-name{font-size:38px}
 .pf-stats{grid-template-columns:repeat(2,1fr)}
}
@media (prefers-reduced-motion:reduce){.pf-ring{animation:none}}
</style>
"""


# ============================================================
# 3. DESIGN CSS PREMIUM
# ============================================================

CSS = """
<style>

:root { --hero-img: url('__HERO_URL__'); }

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #082B49;
    --dark: #103B60;
    --blue: #1769AA;
    --gold: #D5AE61;
    --light: #EAF4FC;
    --background: #F4F8FD;
    --white: #FFFFFF;
    --border: #DCE8F4;
    --muted: #61758B;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, #E5F1FC 0%, transparent 30%),
        linear-gradient(180deg, #F8FBFE, #EDF4FA);
    color: var(--dark);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none;
}

.block-container {
    max-width: 1400px;
    padding: 1.4rem 2.5rem 2rem 2.5rem;
}

h1, h2, h3 {
    color: var(--dark) !important;
    font-weight: 800;
}

h1 {
    letter-spacing: -1px;
}

hr {
    border-color: var(--border);
}

/* MASQUER LES ELEMENTS PAR DEFAUT DE STREAMLIT */

[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNav"],
#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
}

/* LEGENDE DES SCORES */

.score-row {
    display: flex;
    align-items: center;
    gap: 14px;
    background: white;
    border: 1px solid var(--border);
    border-left: 6px solid var(--c);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 5px 15px rgba(20,70,110,.05);
}

.score-row.best {
    border-color: #D5AE61;
    border-left-color: var(--c);
    box-shadow: 0 8px 22px rgba(213,174,97,.25);
}

.score-name {
    color: var(--dark);
    font-weight: 800;
    font-size: 15px;
}

.score-raw {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
}

.score-pct {
    margin-left: auto;
    color: var(--navy);
    font-weight: 800;
    font-size: 24px;
}

/* BANDEAU SUPERIEUR */

.topbar {
    display: flex;
    align-items: center;
    gap: 30px;
    background: rgba(8, 43, 73, .94);
    border: 1px solid rgba(213,174,97,.55);
    border-radius: 18px;
    padding: 18px 28px;
    box-shadow: 0 12px 35px rgba(5,37,65,.25);
}

.topbar-title {
    display: flex;
    align-items: center;
    gap: 14px;
    color: white;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -.5px;
    line-height: 1.1;
}

.topbar-title svg { flex: 0 0 auto; }

.topbar-sub {
    color: #D2E6F5;
    font-size: 11.5px;
    letter-spacing: 1.8px;
    margin-top: 10px;
}

.topbar-gold {
    color: #D5AE61;
    font-size: 11.5px;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin-top: 8px;
}

.topbar-photo {
    flex: 0 0 auto;
    width: 108px;
    height: 108px;
    border-radius: 50%;
    border: 3px solid #D5AE61;
    background: var(--hero-img) 55% 40% / 230% no-repeat, #0A3A63;
    box-shadow: 0 6px 18px rgba(0,0,0,.35);
}

/* MENU (pastilles) */

div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),
div.element-container:has(div[data-testid="stRadio"]) {
    width: 100% !important;
}

div[data-testid="stRadio"] {
    margin-top: 26px;
    margin-bottom: 10px;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

div[data-testid="stRadio"] > div {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

/* Le libellé "Navigation" du widget ne doit jamais s'afficher */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 16px;
    flex-wrap: wrap;
    justify-content: center !important;
    align-items: center;
    width: 100% !important;
    margin: 0 auto;
}

div[data-testid="stRadio"] label {
    background: rgba(8, 43, 73, .95);
    border: 1px solid #3F6E94;
    border-radius: 12px;
    padding: 14px 26px 14px 20px;
    margin: 0;
    min-height: 52px;
    display: flex;
    align-items: center;
    transition: all .25s ease;
    cursor: pointer;
}

div[data-testid="stRadio"] label > div > div:first-child {
    display: none;
}

div[data-testid="stRadio"] label > div {
    display: flex;
    align-items: center;
}

div[data-testid="stRadio"] label > div::before {
    content: "";
    display: block;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: white;
    margin-right: 12px;
    flex: 0 0 auto;
}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: white !important;
    font-size: 15.5px;
    font-weight: 700;
    white-space: nowrap;
}

div[data-testid="stRadio"] label:hover {
    background: #1769AA;
    transform: translateY(-2px);
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: #D5AE61;
    border-color: #D5AE61;
    box-shadow: 0 4px 15px rgba(213,174,97,.35);
}

div[data-testid="stRadio"] label:has(input:checked) > div::before {
    background: radial-gradient(circle, #E5322D 0 38%, white 42%);
    border: 1px solid #A8791F;
}

div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label:has(input:checked) span {
    color: #082B49 !important;
}

/* ACCUEIL : texte de presentation */

.intro-card {
    background: linear-gradient(110deg, #061F38, #0A3A63 60%, #0D4570);
    border: 1px solid rgba(213,174,97,.5);
    border-radius: 22px;
    padding: 38px 42px;
    box-shadow: 0 18px 45px rgba(10,48,80,.15);
}

.intro-card .eyebrow {
    color: #D5AE61;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.intro-card h1 {
    color: white !important;
    font-size: clamp(28px, 3vw, 42px);
    line-height: 1.15;
    margin: 14px 0;
}

.intro-card p {
    color: #EAF5FF;
    font-size: 15px;
    line-height: 1.85;
    max-width: 900px;
}

.tag {
    display: inline-block;
    width: max-content;
    color: white;
    background: rgba(8,43,73,.85);
    border: 1px solid #D5AE61;
    padding: 10px 15px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 800;
}

.hero-spacer {
    display: flex;
    align-items: flex-end;
}

/* CARTES */

.panel, .feature-card, .gallery-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 7px 23px rgba(25,75,120,.06);
}

.feature-card {
    min-height: 190px;
    transition: .3s ease;
}

.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 15px 30px rgba(12,55,90,.13);
}

.feature-icon {
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(135deg,#E7F3FD,#D6EAFB);
    border: 1px solid #C9E0F3;
    font-size: 25px;
    margin-bottom: 14px;
}

.kicker {
    color: #B18A40;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-top: 15px;
}

.gallery-card {
    padding: 12px;
    min-height: 320px;
}

.gallery-title {
    color: #103B60;
    font-size: 16px;
    font-weight: 800;
    margin-top: 12px;
}

.gallery-caption {
    color: #61758B;
    font-size: 12px;
    line-height: 1.6;
    margin-top: 5px;
}

/* INDICATEURS */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-top: 3px solid #D5AE61;
    padding: 19px;
    border-radius: 17px;
    box-shadow: 0 5px 15px rgba(20,70,110,.05);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--dark);
    font-weight: 800;
}

/* BOUTONS */

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg,#1769AA,#2789D4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 800;
    transition: .25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #104F83;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(23,105,170,.25);
}

/* FORMULAIRES */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input {
    background: white !important;
    border: 1px solid #D5E3F0 !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] > div {
    background: white;
    border-radius: 10px;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #9DBDDA;
    border-radius: 13px;
    padding: 10px;
}

.info-box {
    background: #EAF4FF;
    border-left: 5px solid #1769AA;
    border-radius: 13px;
    color: #244E73;
    padding: 17px;
    line-height: 1.8;
}

.warning-box {
    background: #FFF7E8;
    border-left: 5px solid #D49A35;
    border-radius: 13px;
    color: #79551A;
    padding: 17px;
    line-height: 1.8;
}

/* GALERIE : DEFILEMENT AUTOMATIQUE */

.gallery-marquee {
    overflow: hidden;
    position: relative;
    padding: 14px 0 22px 0;
    -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 6%, #000 94%, transparent 100%);
    mask-image: linear-gradient(90deg, transparent 0, #000 6%, #000 94%, transparent 100%);
}

.gallery-track {
    display: flex;
    width: max-content;
    animation: gallery-scroll 50s linear infinite;
    will-change: transform;
}

.gallery-marquee:hover .gallery-track {
    animation-play-state: paused;
}

.gallery-slide {
    flex: 0 0 auto;
    width: 360px;
    margin-right: 24px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 7px 23px rgba(25,75,120,.08);
    transition: transform .3s ease, box-shadow .3s ease;
}

.gallery-slide:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 30px rgba(12,55,90,.15);
}

.gallery-slide img {
    display: block;
    width: 100%;
    height: 240px;
    object-fit: cover;
}

.gallery-info {
    padding: 14px 18px 18px 18px;
    border-top: 3px solid #D5AE61;
}

@keyframes gallery-scroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

@media (max-width: 700px) {
    .gallery-slide { width: 270px; }
    .gallery-slide img { height: 190px; }
}

@media (prefers-reduced-motion: reduce) {
    .gallery-track { animation: none; }
}

/* FOOTER */

.footer {
    background: #082B49;
    color: #D8E8F5;
    text-align: center;
    border: 1px solid #D5AE61;
    border-radius: 18px;
    padding: 28px;
    margin-top: 30px;
    line-height: 2;
    font-size: 12px;
}

</style>
"""

hero_uri = hero_data_uri(HERO_PATH)

CSS = CSS.replace(
    "__HERO_URL__",
    hero_uri
)

st.markdown(CSS, unsafe_allow_html=True)

HOME_CSS = """
<style>
.block-container { position: relative; }

header[data-testid="stHeader"] { background: transparent; }

.block-container::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: min(calc(min(100vw - 15px, 1400px) * 0.671), 860px);
    border-radius: 0 0 26px 26px;
    background: var(--hero-img) center 20% / cover no-repeat, linear-gradient(120deg,#0A3A63,#104B75);
    box-shadow: 0 18px 45px rgba(10,48,80,.18);
    z-index: 0;
}

.block-container > div { position: relative; z-index: 1; }

.hero-spacer {
    height: max(60px, calc(min(calc(min(100vw - 15px, 1400px) * 0.671), 860px) - 285px));
}

@media (max-width: 900px) {
    .hero-spacer { height: 40px; }
    .block-container::before { height: 590px; }
    .topbar { flex-wrap: wrap; }
    .topbar-title { font-size: 26px; }
}
</style>
"""

# Lien direct (QR code) : https://votre-app/?page=profil
try:
    if (
        st.query_params.get("page", "").lower() == "profil"
        and not st.session_state.get("_deeplink_done")
    ):
        st.session_state["navigation"] = "Profil"
        st.session_state["_deeplink_done"] = True
except Exception:
    pass

if st.session_state.get("navigation", "Accueil") == "Accueil":
    st.markdown(HOME_CSS, unsafe_allow_html=True)


# ============================================================
# 4. SESSION STATE
# ============================================================

defaults = {
    "page": "Accueil",
    "navigation": "Accueil",
    "records": [],
    "last": None,
    "model_file": None,
    "model_hash": None,
    "gallery_index": 0,
    "class_names_text": (
        "Tumeur bénigne\n"
        "Tumeur maligne\n"
        "Image normale"
    )
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 5. CHARGEMENT DU MODELE
# ============================================================

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(
        path,
        compile=False
    )


def prepare_model(uploaded_model, local_path):

    if uploaded_model is not None:

        content = uploaded_model.getvalue()

        file_hash = hashlib.sha256(
            content
        ).hexdigest()

        if st.session_state.model_hash != file_hash:

            suffix = (
                os.path.splitext(uploaded_model.name)[1]
                or ".keras"
            )

            temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            temp.write(content)
            temp.close()

            st.session_state.model_file = temp.name
            st.session_state.model_hash = file_hash

        return st.session_state.model_file

    if (
        st.session_state.model_file
        and os.path.exists(st.session_state.model_file)
    ):
        return st.session_state.model_file

    if local_path and os.path.exists(local_path):
        return local_path

    return None


# ============================================================
# 6. PRETRAITEMENT DES IMAGES
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    original = np.asarray(
        image,
        dtype=np.uint8
    )

    batch = preprocess_input(
        original.astype(np.float32).copy()
    )

    batch = np.expand_dims(
        batch,
        axis=0
    )

    return batch, original


def build_report_html(row):

    def v(key):
        value = row.get(key, "—")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = "—"
        return html_lib.escape(str(value))

    fields = [
        ("Identifiant", "ID"), ("Patient (code)", "Patient"),
        ("Âge", "Âge"), ("Date de l'examen", "Date examen"),
        ("Médecin / opérateur", "Médecin"),
        ("Classe donnée par le modèle", "Classe modèle"),
        ("Score du modèle (%)", "Score modèle (%)"),
        ("Statut de revue", "Avis professionnel"),
        ("Commentaire du professionnel", "Commentaire"),
        ("Horodatage", "Horodatage"),
    ]

    rows = "".join(
        f"<tr><th>{label}</th><td>{v(key)}</td></tr>"
        for label, key in fields
    )

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Rapport {v("ID")}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:800px;margin:30px auto;color:#103B60}}
h1{{border-bottom:3px solid #D5AE61;padding-bottom:8px}}
table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:9px;border:1px solid #DCE8F4}}
th{{background:#EAF4FC;width:38%}}
.note{{margin-top:24px;background:#FFF7E8;border-left:5px solid #D49A35;padding:14px}}
</style></head><body>
<h1>IA Soins du Sein — Rapport d'analyse expérimentale</h1>
<table>{rows}</table>
<div class="note">Prototype académique. Le score est une sortie du modèle,
pas une probabilité clinique validée ni un diagnostic médical. Ce document
doit être interprété par un professionnel de santé qualifié.</div>
</body></html>"""


# ============================================================
# 7. GRAD-CAM POUR VGG16
# ============================================================

def find_backbone(model):

    for i, layer in enumerate(model.layers):

        if isinstance(layer, tf.keras.Model):

            try:
                layer.get_layer("block5_conv3")
                return i, layer

            except ValueError:
                continue

    raise ValueError(
        "VGG16 imbriqué avec block5_conv3 introuvable. "
        "Vérifiez l'architecture de votre modèle."
    )


def gradcam(model, batch, class_index):

    if not model.layers:
        raise ValueError("Le modèle ne contient aucune couche.")

    backbone_index, backbone = find_backbone(model)

    conv_layer = backbone.get_layer("block5_conv3")

    grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            conv_layer.output,
            backbone.output
        ]
    )

    head_layers = model.layers[backbone_index + 1:]

    inputs = tf.convert_to_tensor(
        batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        activations, features = grad_model(
            inputs,
            training=False
        )

        x = features

        for layer in head_layers:
            x = layer(
                x,
                training=False
            )

        score = x[:, class_index]

    gradients = tape.gradient(
        score,
        activations
    )

    if gradients is None:
        raise ValueError(
            "Gradients indisponibles. "
            "Vérifiez l'architecture du modèle."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    heatmap = tf.reduce_sum(
        activations[0] * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy().astype(np.float32)


def create_overlay(original, heatmap):

    h, w = original.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (w, h)
    )

    heat_uint8 = np.uint8(
        np.clip(heatmap * 255, 0, 255)
    )

    colored = cv2.cvtColor(
        cv2.applyColorMap(
            heat_uint8,
            cv2.COLORMAP_JET
        ),
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored,
        0.4,
        0
    )

    return heatmap, overlay


# ============================================================
# 8. BARRE DE NAVIGATION PREMIUM
# ============================================================

pages = [
    "Accueil",
    "Tableau de bord",
    "Analyse IA",
    "Patients",
    "Rapports",
    "À propos",
    "Profil"
]

PAGE_LABELS = {
    "Accueil": "Accueil",
    "Tableau de bord": "Tableau de bord",
    "Analyse IA": "Analyse IA",
    "Patients": "Les patients",
    "Rapports": "Rapports",
    "À propos": "À propos de",
    "Profil": "Profil"
}

render("""
<div class="topbar">
    <div>
        <div class="topbar-title">
            <svg width="38" height="38" viewBox="0 0 24 24" fill="none"
                 stroke="#8E7BFF" stroke-width="1.8" stroke-linecap="round"
                 stroke-linejoin="round">
                <path d="M6 3v6a4 4 0 0 0 8 0V3"/>
                <path d="M10 13v2a5 5 0 0 0 10 0v-1"/>
                <circle cx="20" cy="12" r="2" stroke="#4FC3F7"/>
                <path d="M5 3h2M13 3h2"/>
            </svg>
            IA SOINS DU SEIN
        </div>
        <div class="topbar-sub">
            INTELLIGENCE ARTIFICIELLE · SANTÉ · IMAGERIE MAMMAIRE
        </div>
        <div class="topbar-gold">
            INNOVATION MÉDICALE · APPRENTISSAGE PROFOND · GRAD-CAM
        </div>
    </div>
    <div class="topbar-photo"></div>
</div>
""")

selected_page = st.radio(
    "Navigation",
    pages,
    format_func=lambda p: PAGE_LABELS[p],
    horizontal=True,
    label_visibility="collapsed",
    key="navigation"
)

st.session_state.page = selected_page

if selected_page == "Accueil":
    render("""<div class="hero-spacer">
        <span class="tag">VGG16 · CNN · DEEP LEARNING · GRAD-CAM</span>
    </div>""")
    if not hero_uri:
        st.warning("Image d'accueil introuvable : placez hero.jpg à côté de app.py.")
else:
    st.markdown("---")


# ============================================================
# 9. CONFIGURATION DU MODELE IA
# ============================================================

model = None

with st.expander(
    "⚙️ Configuration du modèle d'intelligence artificielle",
    expanded=False
):

    col1, col2, col3 = st.columns([1.5, 1.2, 1.5])

    with col1:

        uploaded_model = st.file_uploader(
            "Importer le modèle VGG16",
            type=["keras", "h5"],
            key="model_uploader"
        )

    with col2:

        local_path = st.text_input(
            "Chemin du modèle",
            value="modele_vgg16.keras"
        )

    with col3:

        st.session_state.class_names_text = st.text_area(
            "Noms des trois classes",
            value=st.session_state.class_names_text,
            height=110
        )

    model_path = prepare_model(
        uploaded_model,
        local_path
    )

    if model_path:

        try:

            model = load_model(model_path)

            st.success(
                "Modèle chargé avec succès."
            )

        except Exception as e:

            st.error(
                f"Erreur de chargement : {e}"
            )

    else:

        st.info(
            "Importez votre modèle .keras ou .h5 "
            "ou placez modele_vgg16.keras à côté de app.py."
        )


class_names = [
    x.strip()
    for x in st.session_state.class_names_text.splitlines()
    if x.strip()
]

if len(class_names) != 3:

    st.warning(
        "Veuillez saisir exactement trois classes."
    )

    class_names = [
        "Tumeur bénigne",
        "Tumeur maligne",
        "Image normale"
    ]

page = st.session_state.page


# ============================================================
# 10. PAGE ACCUEIL
# ============================================================

if page == "Accueil":

    st.write("")

    render("""
    <div class="intro-card">
        <div class="eyebrow">BreastCare AI · INNOVATION EN SANTÉ</div>
        <h1>L'intelligence artificielle au service de l'imagerie mammaire.</h1>
        <p>
            Cette plateforme expérimentale est dédiée à l'analyse des
            images échographiques mammaires. Elle exploite le Deep Learning,
            l'architecture VGG16 et la visualisation Grad-CAM pour explorer
            la classification des images en trois catégories : tumeurs
            bénignes, tumeurs malignes et tissus normaux.
        </p>
    </div>
    """)

    st.write("")

    st.markdown(
        '<div class="kicker">ESPACE DE TRAVAIL</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Une plateforme dédiée à l'analyse expérimentale"
    )

    a, b, c = st.columns(3)

    with a:

        render("""
        <div class="feature-card">
            <div class="feature-icon">👩‍⚕️</div>
            <h3>Espace médical</h3>
            <p>
                Importer une image, examiner les résultats
                et documenter un avis professionnel.
            </p>
        </div>
        """)

    with b:

        render("""
        <div class="feature-card">
            <div class="feature-icon">🗂️</div>
            <h3>Dossiers patients</h3>
            <p>
                Créer des profils anonymisés et consulter
                les analyses enregistrées.
            </p>
        </div>
        """)

    with c:

        render("""
        <div class="feature-card">
            <div class="feature-icon">🔥</div>
            <h3>Explicabilité Grad-CAM</h3>
            <p>
                Visualiser les régions ayant contribué
                aux prédictions du modèle.
            </p>
        </div>
        """)

    # --------------------------------------------------------
    # GALERIE MEDICALE
    # --------------------------------------------------------

    st.write("")

    st.markdown(
        '<div class="kicker">ENVIRONNEMENT MÉDICAL</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Consultation, recherche et accompagnement"
    )

    st.write(
        "Une approche centrée sur la patiente, "
        "la collaboration médicale et l'innovation technologique."
    )

    medical_images = [
        (
            "gallery_1.jpg",
            "Innovation médicale",
            "Technologies numériques au service de la santé."
        ),
        (
            "gallery_2.jpg",
            "Consultation médicale",
            "Dialogue et accompagnement personnalisé des patientes."
        ),
        (
            "gallery_3.jpg",
            "Équipe médicale",
            "Collaboration entre professionnels de santé."
        ),
        (
            "gallery_4.jpg",
            "Accompagnement des patientes",
            "Une prise en charge humaine et attentive."
        ),
        (
            "gallery_5.jpg",
            "Recherche médicale",
            "Exploration de nouvelles approches diagnostiques."
        )
    ]

    slides = []

    for filename, title, caption in medical_images:

        uri = gallery_data_uri(BASE_DIR / filename)

        if uri:
            slides.append(
                '<div class="gallery-slide">'
                f'<img src="{uri}" alt="{html_lib.escape(title)}">'
                '<div class="gallery-info">'
                f'<div class="gallery-title">{html_lib.escape(title)}</div>'
                f'<div class="gallery-caption">{html_lib.escape(caption)}</div>'
                '</div></div>'
            )

    if slides:

        # Les diapositives sont dupliquées pour obtenir une boucle sans coupure
        track = "".join(slides) * 2

        st.markdown(
            '<div class="gallery-marquee">'
            f'<div class="gallery-track">{track}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Images de la galerie introuvables : placez "
            "gallery_1.jpg à gallery_5.jpg à côté de apps.py."
        )

    # --------------------------------------------------------
    # ENGAGEMENTS
    # --------------------------------------------------------

    st.write("")

    st.markdown(
        '<div class="kicker">NOTRE VISION</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Une innovation technologique au service de la santé"
    )

    st.write(
        "BreastCare AI explore le potentiel de l'intelligence "
        "artificielle pour accompagner la recherche et l'analyse "
        "des images échographiques mammaires."
    )

    c1, c2, c3 = st.columns(3)

    engagements = [

        (
            "🔬",
            "Innovation scientifique",
            "Explorer les capacités du Deep Learning appliqué à l'imagerie médicale."
        ),

        (
            "🧠",
            "Intelligence artificielle",
            "Développer une approche expérimentale de classification des images mammaires."
        ),

        (
            "🤝",
            "Accompagnement médical",
            "Favoriser une approche collaborative entre technologie et expertise médicale."
        )
    ]

    for col, (icon, title, description) in zip(
        [c1, c2, c3],
        engagements
    ):

        with col:

            render(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """)

    render("""
    <div class="info-box">
        <b>Information importante :</b>
        BreastCare AI est un prototype académique.
        Les résultats sont expérimentaux et ne constituent
        pas un diagnostic médical.
    </div>
    """)


# ============================================================
# 11. PAGE ANALYSE IA
# ============================================================

elif page == "Analyse IA":

    st.title(
        "🔬 Analyse intelligente des images mammaires"
    )

    st.caption(
        "Importation · Classification · Grad-CAM · Revue professionnelle"
    )

    if model is None:

        st.warning(
            "Veuillez charger votre modèle dans "
            "Configuration du modèle IA."
        )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "01. Informations patient"
        )

        pid = st.text_input(
            "Identifiant patient *",
            placeholder="BC-2026-001"
        )

        pname = st.text_input(
            "Code anonymisé",
            placeholder="Patiente A."
        )

        age = st.number_input(
            "Âge",
            0,
            120,
            35
        )

        exam_date = st.date_input(
            "Date de l'examen",
            date.today()
        )

        doctor = st.text_input(
            "Médecin / opérateur"
        )

    with right:

        st.subheader(
            "02. Image échographique"
        )

        uploaded_image = st.file_uploader(
            "Importer une image mammaire",
            type=["png", "jpg", "jpeg"],
            key="image_uploader"
        )

        pil_image = None

        if uploaded_image:

            pil_image = Image.open(
                uploaded_image
            ).convert("RGB")

            st.image(
                pil_image,
                caption="Image importée",
                **FIT
            )

    if st.button(
        "🚀 Lancer l'analyse expérimentale",
        type="primary",
        disabled=(
            model is None or uploaded_image is None
        ),
        **FIT
    ):

        if not pid.strip():

            st.error(
                "Identifiant patient obligatoire."
            )

        else:

            try:

                batch, original = prepare_image(
                    pil_image
                )

                predictions = model.predict(
                    batch,
                    verbose=0
                )[0]

                if len(predictions) != len(class_names):

                    raise ValueError(
                        "Le nombre de classes ne correspond "
                        "pas à la sortie du modèle."
                    )

                idx = int(
                    np.argmax(predictions)
                )

                heatmap = gradcam(
                    model,
                    batch,
                    idx
                )

                heatmap, overlay = create_overlay(
                    original,
                    heatmap
                )

                st.session_state.last = {

                    "ID": pid.strip(),

                    "Patient": pname or "Non renseigné",

                    "Âge": int(age),

                    "Date examen": str(exam_date),

                    "Médecin": doctor or "Non renseigné",

                    "Classe modèle": class_names[idx],

                    "Score modèle (%)": round(
                        float(predictions[idx]) * 100,
                        2
                    ),

                    "probabilities": predictions.tolist(),

                    "image": original,

                    "heatmap": heatmap,

                    "overlay": overlay,

                    "Horodatage": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
                }

                st.success(
                    "Analyse expérimentale terminée."
                )

            except Exception as e:

                st.error(
                    f"Erreur pendant l'analyse : {e}"
                )

    result = st.session_state.last

    if result:

        st.markdown("---")

        st.subheader(
            "03. Résultats du modèle"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Classe prédite",
            result["Classe modèle"]
        )

        b.metric(
            "Score du modèle",
            f'{result["Score modèle (%)"]:.2f}%'
        )

        c.metric(
            "Identifiant",
            result["ID"]
        )

        render("""
        <div class="warning-box">
            Le score affiché correspond à une sortie du modèle.
            Il ne constitue pas une probabilité clinique validée
            ni un diagnostic médical.
        </div>
        """)

        st.subheader(
            "Distribution des scores"
        )

        percents = [
            float(p) * 100
            for p in result["probabilities"]
        ]

        chart_col, legend_col = st.columns([3, 2])

        with chart_col:

            show_vega(
                score_donut_spec(class_names, percents)
            )

        with legend_col:

            st.write("")
            st.write("")

            best_index = int(np.argmax(percents))

            rows = ""

            for i, (name, pct) in enumerate(
                zip(class_names, percents)
            ):

                raw = pct / 100
                best = " best" if i == best_index else ""

                rows += (
                    f'<div class="score-row{best}" '
                    f'style="--c:{SCORE_COLORS[i]}">'
                    '<div>'
                    f'<div class="score-name">{html_lib.escape(name)}</div>'
                    f'<div class="score-raw">Valeur : {raw:.4f}</div>'
                    '</div>'
                    f'<div class="score-pct">{pct:.1f} %</div>'
                    '</div>'
                )

            st.markdown(rows, unsafe_allow_html=True)

        st.subheader(
            "04. Explicabilité Grad-CAM"
        )

        c1, c2, c3 = st.columns(3)

        c1.image(
            result["image"],
            caption="Image originale",
            **FIT
        )

        c2.image(
            result["heatmap"],
            caption="Carte thermique",
            clamp=True,
            **FIT
        )

        c3.image(
            result["overlay"],
            caption="Superposition Grad-CAM",
            **FIT
        )

        st.subheader(
            "05. Avis professionnel"
        )

        status = st.selectbox(
            "Statut de revue",
            [
                "En attente de revue médicale",
                "À corréler avec l'examen clinique",
                "Examen complémentaire à discuter",
                "Avis documenté par le médecin"
            ]
        )

        note = st.text_area(
            "Commentaire du professionnel"
        )

        if st.button(
            "💾 Enregistrer le dossier"
        ):

            record = {
                k: v for k, v in result.items()
                if k not in [
                    "probabilities",
                    "image",
                    "heatmap",
                    "overlay"
                ]
            }

            record["Avis professionnel"] = status
            record["Commentaire"] = note

            existing = next(
                (
                    p for p in st.session_state.records
                    if p["ID"] == record["ID"]
                ),
                None
            )

            if existing is not None:

                existing.update(record)

            else:

                st.session_state.records.append(
                    record
                )

            st.success(
                "Dossier enregistré avec succès."
            )


# ============================================================
# 12. PAGE TABLEAU DE BORD
# ============================================================

elif page == "Tableau de bord":

    st.title(
        "📊 Tableau de bord hospitalier"
    )

    st.caption(
        "Suivi des dossiers et synthèse des analyses expérimentales"
    )

    df = pd.DataFrame(
        st.session_state.records
    )

    total = len(df)

    pending = 0

    if total and "Avis professionnel" in df:

        pending = int(
            df["Avis professionnel"]
            .astype(str)
            .str.contains(
                "attente",
                case=False,
                na=False
            )
            .sum()
        )

    reviewed = total - pending

    average = None

    if (
        total
        and "Score modèle (%)" in df
        and df["Score modèle (%)"].notna().any()
    ):

        average = df["Score modèle (%)"].mean()

    a, b, c, d = st.columns(4)

    a.metric(
        "Dossiers enregistrés",
        total
    )

    b.metric(
        "Dossiers revus",
        reviewed
    )

    c.metric(
        "En attente",
        pending
    )

    d.metric(
        "Score moyen IA",
        f"{average:.1f}%" if average is not None else "—"
    )

    if total:

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Répartition des classes"
            )

            if "Classe modèle" in df:

                st.bar_chart(
                    df["Classe modèle"].value_counts()
                )

        with right:

            st.subheader(
                "Statut des dossiers"
            )

            if "Avis professionnel" in df:

                st.bar_chart(
                    df["Avis professionnel"].value_counts()
                )

        st.subheader(
            "Derniers dossiers"
        )

        st.dataframe(
            df,
            hide_index=True,
            **FIT
        )

    else:

        st.info(
            "Aucun dossier enregistré pour le moment."
        )


# ============================================================
# 13. PAGE PATIENTS
# ============================================================

elif page == "Patients":

    st.title(
        "👥 Gestion des profils patients"
    )

    st.caption(
        "Création et consultation des dossiers anonymisés"
    )

    with st.form("new_patient"):

        a, b = st.columns(2)

        with a:

            pid = st.text_input(
                "Identifiant unique *"
            )

            name = st.text_input(
                "Code anonymisé"
            )

            age = st.number_input(
                "Âge",
                0,
                120,
                30
            )

        with b:

            sex = st.selectbox(
                "Sexe",
                [
                    "Femme",
                    "Homme",
                    "Autre / non renseigné"
                ]
            )

            created = st.date_input(
                "Date de création",
                date.today()
            )

        submit = st.form_submit_button(
            "＋ Créer le profil",
            **FIT
        )

        if submit:

            if not pid.strip():

                st.error(
                    "Identifiant obligatoire."
                )

            elif any(
                x.get("ID") == pid.strip()
                for x in st.session_state.records
            ):

                st.error(
                    "Identifiant déjà présent."
                )

            else:

                st.session_state.records.append({

                    "ID": pid.strip(),

                    "Patient": name or "Non renseigné",

                    "Âge": int(age),

                    "Sexe": sex,

                    "Date création": str(created),

                    "Classe modèle": "Aucune analyse",

                    "Score modèle (%)": np.nan,

                    "Avis professionnel": (
                        "Profil créé — analyse en attente"
                    ),

                    "Commentaire": "",

                    "Horodatage": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
                })

                st.success(
                    "Profil créé avec succès."
                )

    df = pd.DataFrame(
        st.session_state.records
    )

    if len(df):

        st.subheader(
            "Registre des patients"
        )

        st.dataframe(
            df,
            hide_index=True,
            **FIT
        )

        chosen = st.selectbox(
            "Ouvrir un dossier",
            df["ID"].tolist()
        )

        row = df[
            df["ID"] == chosen
        ].iloc[-1]

        st.subheader(
            f"Dossier {chosen}"
        )

        a, b = st.columns(2)

        with a:

            st.write(
                "**Patient :**",
                row.get("Patient", "—")
            )

            st.write(
                "**Âge :**",
                row.get("Âge", "—")
            )

            st.write(
                "**Classe :**",
                row.get("Classe modèle", "—")
            )

        with b:

            st.write(
                "**Avis professionnel :**",
                row.get("Avis professionnel", "—")
            )

            st.write(
                "**Commentaire :**",
                row.get("Commentaire", "—")
            )

    else:

        st.info(
            "Aucun profil patient enregistré."
        )


# ============================================================
# 14. PAGE RAPPORTS
# ============================================================

elif page == "Rapports":

    st.title(
        "📄 Rapports et exportation"
    )

    st.caption(
        "Consultation et exportation des données enregistrées"
    )

    df = pd.DataFrame(
        st.session_state.records
    )

    if len(df):

        st.subheader(
            "Registre des analyses"
        )

        st.dataframe(
            df,
            hide_index=True,
            **FIT
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "⬇ Télécharger les dossiers CSV",
            data=csv,
            file_name="breastcare_patients.csv",
            mime="text/csv",
            **FIT
        )

        st.subheader(
            "Rapport individuel"
        )

        report_id = st.selectbox(
            "Choisir un dossier",
            df["ID"].tolist(),
            key="report_choice"
        )

        report_row = df[
            df["ID"] == report_id
        ].iloc[-1].to_dict()

        st.download_button(
            "⬇ Télécharger le rapport (HTML imprimable)",
            data=build_report_html(report_row).encode("utf-8"),
            file_name=f"rapport_{report_id}.html",
            mime="text/html",
            **FIT
        )

    else:

        st.info(
            "Aucun dossier à exporter."
        )


# ============================================================
# 15. PAGE A PROPOS
# ============================================================

elif page == "À propos":

    st.title(
        "À propos de BreastCare AI"
    )

    st.caption(
        "Présentation scientifique et technologique du projet"
    )

    render("""
    <div class="panel">

        <div class="kicker">
            PRÉSENTATION SCIENTIFIQUE
        </div>

        <h3>Objectif du projet</h3>

        <p>
            Développer un prototype académique de classification
            des images échographiques mammaires par réseau de
            neurones convolutif VGG16 et méthode d'explicabilité
            Grad-CAM.
        </p>

        <hr>

        <h3>Technologies utilisées</h3>

        <p>
            Python · TensorFlow · Keras · VGG16 · OpenCV ·
            NumPy · Pandas · Streamlit
        </p>

        <hr>

        <h3>Fonctionnalités principales</h3>

        <ul>
            <li>Importation du modèle entraîné.</li>
            <li>Prétraitement des images en 224 × 224.</li>
            <li>Classification en trois catégories.</li>
            <li>Visualisation Grad-CAM.</li>
            <li>Gestion des profils anonymisés.</li>
            <li>Tableau de bord et export CSV.</li>
        </ul>

        <hr>

        <h3>Limites et sécurité</h3>

        <ul>
            <li>Prototype non validé comme dispositif médical.</li>
            <li>Les scores ne sont pas des probabilités cliniques validées.</li>
            <li>Grad-CAM est une explication visuelle approximative.</li>
            <li>Ne pas utiliser pour diagnostiquer ou traiter une patiente.</li>
            <li>Les données doivent être anonymisées.</li>
            <li>Les dossiers sont conservés dans la session Streamlit.</li>
        </ul>

    </div>
    """)


# ============================================================
# 15b. PAGE PROFIL (recruteurs)
# ============================================================

elif page == "Profil":

    p = PROFILE
    st.markdown(PROFILE_CSS, unsafe_allow_html=True)

    photo_uri = hero_data_uri(PROFILE_PHOTO)

    if photo_uri:
        photo_html = f'<img src="{photo_uri}" alt="Ibrahima DIENG">'
    else:
        photo_html = '<div class="pf-initials">ID</div>'
        st.warning(
            "Photo introuvable : placez profil_ibrahima.jpg "
            "à côté de app.py."
        )

    # ---------- En-tête ----------
    render(f"""
    <div class="pf-hero">
        <div class="pf-photo">
            <div class="pf-ring"></div>
            {photo_html}
        </div>
        <div class="pf-id">
            <div class="pf-kicker">PROFIL PROFESSIONNEL · DATA SCIENCE &amp; IA</div>
            <div class="pf-name">IBRAHIMA <span>DIENG</span></div>
            <div class="pf-title">{_e(p['title'])}</div>
            <div class="pf-badge">M1 DATA SCIENCE &amp; IA · DAKAR INSTITUTE OF TECHNOLOGY</div>
            <div class="pf-chips">
                <a class="pf-chip" href="tel:{_e(p['phone_raw'])}">📞 {_e(p['phone'])}</a>
                <a class="pf-chip" href="mailto:{_e(p['email'])}">✉️ {_e(p['email'])}</a>
                <a class="pf-chip" href="{_e(p['github_url'])}" target="_blank" rel="noopener">🔗 {_e(p['github'])}</a>
                <span class="pf-chip">📍 {_e(p['location'])}</span>
            </div>
        </div>
    </div>
    """)

    # ---------- Chiffres clés ----------
    n_tools = len({i for _, items in p["skills"] for i in items})

    render(f"""
    <div class="pf-stats">
        <div class="pf-stat"><b>Licence SID</b><span>DIPLÔMÉ · 2023</span></div>
        <div class="pf-stat"><b>Master 1</b><span>ÉQUIVALENT · EN COURS</span></div>
        <div class="pf-stat"><b>{len(p['projects'])} domaines</b><span>PROJETS ACADÉMIQUES</span></div>
        <div class="pf-stat"><b>{n_tools}</b><span>OUTILS &amp; TECHNOLOGIES</span></div>
    </div>
    """)

    # ---------- Contenu du CV ----------
    left, right = st.columns([1.55, 1], gap="large")

    with left:

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">👤</span>Résumé professionnel</div>
            <div class="pf-sum">{_e(p['summary'])}</div>
        </div>
        """)

        projects_html = "".join(
            f'<div class="pf-proj"><b>{_e(t)}</b><ul>'
            + "".join(f"<li>{_e(i)}</li>" for i in items)
            + "</ul></div>"
            for t, items in p["projects"]
        )

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">🧪</span>Projets académiques</div>
            {projects_html}
        </div>
        """)

        edu_html = "".join(
            f'<div class="pf-tl-i{" now" if now else ""}">'
            f'<div class="pf-tl-t">{_e(t)}</div>'
            f'<div class="pf-tl-s">{_e(s)}</div>'
            f'<span class="pf-tl-d">{_e(d)}</span></div>'
            for t, s, d, now in p["education"]
        )

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">🎓</span>Formation</div>
            <div class="pf-tl">{edu_html}</div>
        </div>
        """)

    with right:

        skills_html = "".join(
            f'<div class="pf-sk-t">{_e(t)}</div>'
            + "".join(f'<span class="pf-tag">{_e(i)}</span>' for i in items)
            for t, items in p["skills"]
        )

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">⚙️</span>Compétences techniques</div>
            {skills_html}
        </div>
        """)

        cert_html = "".join(
            f'<div class="pf-tl-i{" now" if now else ""}">'
            f'<div class="pf-tl-t">{_e(t)}</div>'
            f'<div class="pf-tl-s">{_e(s)}</div>'
            f'<span class="pf-tl-d">{_e(d)}</span></div>'
            for t, s, d, now in p["certs"]
        )

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">🏅</span>Certifications</div>
            <div class="pf-tl">{cert_html}</div>
        </div>
        """)

        community_html = "".join(
            f"<li>{_e(i)}</li>" for i in p["community"]
        )

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">🤝</span>Engagement social</div>
            <div class="pf-org">{_e(p['community_org'])}</div>
            <div class="pf-role">{_e(p['community_role']).upper()}</div>
            <ul class="pf-list">{community_html}</ul>
        </div>
        """)

        langs_html = "".join(
            f'<span class="pf-tag">{_e(i)}</span>' for i in p["languages"]
        )
        interests_html = "".join(
            f'<span class="pf-tag">{_e(i)}</span>' for i in p["interests"]
        )

        render(f"""
        <div class="pf-card">
            <div class="pf-h"><span class="pf-ico">🌍</span>Langues &amp; intérêts</div>
            <div class="pf-sk-t">LANGUES</div>{langs_html}
            <div class="pf-sk-t">CENTRES D'INTÉRÊT</div>{interests_html}
        </div>
        """)

    # ---------- Scanner recruteur ----------
    render("""
    <div class="pf-scan">
        <div class="pf-kicker">SCANNER RECRUTEUR</div>
        <div class="pf-scan-t">Scannez le profil, retrouvez le CV</div>
        <p>
            Le recruteur pointe l'appareil photo de son téléphone vers un
            QR code : il enregistre directement le contact ou ouvre le profil
            en ligne.
        </p>
    </div>
    """)

    profile_url = st.text_input(
        "Lien encodé dans le QR code « Profil en ligne »",
        value=st.session_state.get("pf_url", p["github_url"]),
        key="pf_url",
        help=(
            "Une fois l'application déployée, collez ici son adresse "
            "suivie de ?page=profil : le QR code ouvrira directement "
            "cette page."
        )
    ).strip() or p["github_url"]

    qr_contact = make_qr_uri(build_vcard())
    qr_online = make_qr_uri(profile_url, dark="#1769AA")

    if not qr_contact:
        st.warning(
            "Le module de QR code est absent : installez-le avec "
            "« pip install segno » puis relancez l'application."
        )
    else:
        qa, qb = st.columns(2, gap="large")

        with qa:
            render(f"""
            <div class="pf-qr">
                <img src="{qr_contact}" alt="QR code contact">
                <b>Carte de contact</b>
                <span>Fonctionne hors ligne : nom, téléphone, e-mail et GitHub
                s'ajoutent aux contacts du téléphone.</span>
            </div>
            """)

        with qb:
            render(f"""
            <div class="pf-qr">
                <img src="{qr_online}" alt="QR code profil en ligne">
                <b>Profil en ligne</b>
                <span>{_e(profile_url)}</span>
            </div>
            """)

    st.write("")

    cv_html = build_cv_html(photo_uri, qr_online)

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.download_button(
            "📄 Télécharger le CV (imprimable en PDF)",
            data=cv_html.encode("utf-8"),
            file_name="CV_Ibrahima_DIENG.html",
            mime="text/html",
            **FIT
        )

    with d2:
        st.download_button(
            "📇 Enregistrer le contact (.vcf)",
            data=build_vcard().encode("utf-8"),
            file_name="Ibrahima_DIENG.vcf",
            mime="text/vcard",
            **FIT
        )



# ============================================================
# 16. FOOTER INSTITUTIONNEL
# ============================================================

st.markdown("---")

render("""
<div class="footer">

    <div style="font-size:26px;font-weight:800;color:white;">
        🩺 BreastCare AI  
    </div>

    <div style="color:#D5AE61;font-size:11px;
                letter-spacing:2px;font-weight:800;">
        INTELLIGENCE ARTIFICIELLE · IMAGERIE MAMMAIRE
    </div>

    <p>
        Une approche expérimentale de l'intelligence
        artificielle appliquée à l'analyse des images
        échographiques mammaires.
    </p>

    <hr style="border-color:#31536F;">

    <p>
        PYTHON · TENSORFLOW · VGG16 · CNN · GRAD-CAM · STREAMLIT
    </p>

    <p>
        Prototype académique de recherche et d'innovation.
        Les résultats ne constituent pas un diagnostic médical
        et nécessitent une évaluation clinique indépendante.
    </p>

    <div style="color:#D5AE61;font-weight:800;">
        BreastCare AI — TECHNOLOGIE ET INNOVATION EN SANTÉ
    </div>

</div>
""")
