import os
import time
import io
import json
import base64
import colorsys
import traceback

import pandas as pd
import numpy as np
import networkx as nx
import cv2
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models, transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# Support optionnel GeoTIFF
try:
    import rasterio
    from rasterio.plot import reshape_as_image
except ImportError:
    rasterio = None


# =====================================================
# 1. CONFIGURATION
# =====================================================
load_dotenv()

# Client Puter API Bridge (serveur local, aucune clé requise)
client = OpenAI(
    api_key="unused",
    base_url="http://127.0.0.1:8741/v1",
)

MODEL_TEXT = "gpt-4o"
MODEL_VISION = "gpt-4o"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# 1.bis CHARGEMENT DU MODÈLE DeepLabV3 (segmentation)
# =====================================================
_WEIGHTS = None
_MODEL_IA = None
_PREPROCESS = None


def get_segmentation_model():
    """Charge (une seule fois) le modèle DeepLabV3."""
    global _WEIGHTS, _MODEL_IA, _PREPROCESS
    if _MODEL_IA is None:
        _WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT
        _MODEL_IA = deeplabv3_resnet50(weights=_WEIGHTS).to(DEVICE).eval()
        _PREPROCESS = _WEIGHTS.transforms()
    return _MODEL_IA, _PREPROCESS, _WEIGHTS


# =====================================================
# 1.ter CHARGEMENT DU MODÈLE CNN (classification sols)
# =====================================================
_SOIL_MODEL = None
_SOIL_CLASSES = None
_SOIL_TRANSFORM = None


def get_soil_classifier():
    """Charge le modèle CNN de classification des sols (une seule fois)."""
    global _SOIL_MODEL, _SOIL_CLASSES, _SOIL_TRANSFORM

    if _SOIL_MODEL is None:
        chemin_modele = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "soil_model.pth",
        )

        if not os.path.exists(chemin_modele):
            raise FileNotFoundError(
                f"Modèle introuvable : {chemin_modele}. "
                f"Lance d'abord 'python analyser/train_soil.py'"
            )

        checkpoint = torch.load(chemin_modele, map_location=DEVICE)

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features, checkpoint["num_classes"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(DEVICE).eval()

        _SOIL_MODEL = model
        _SOIL_CLASSES = checkpoint["classes"]
        _SOIL_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    return _SOIL_MODEL, _SOIL_CLASSES, _SOIL_TRANSFORM


# =====================================================
# 2. APPELS LLM — TEXTE & VISION (Puter API Bridge)
# =====================================================
def generer_texte(prompt, system_prompt="Tu es un assistant IA expert.", retries=1):
    """Appel texte via Puter API Bridge (aucune clé requise)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    for tentative in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                timeout=90,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Puter Text Error] Tentative {tentative + 1}/{retries} : {e}")
            time.sleep(2)

    return "Erreur : Le service Puter est temporairement indisponible."


def generer_vision(image_pil, prompt, retries=1):
    """Appel vision via Puter API Bridge."""
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]

    for tentative in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_VISION,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
                timeout=90,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Puter Vision Error] Tentative {tentative + 1}/{retries} : {e}")
            time.sleep(2)

    return "Erreur : Le service Puter Vision est temporairement indisponible."


# =====================================================
# 2.bis CLASSIFICATION D'IMAGE
# =====================================================
def classifier_image(image_pil):
    """
    Demande à l'IA de classifier l'image avant tout traitement.
    Retourne : 'satellite', 'sol', 'medical', 'paysage', 'urbain' ou 'autre'.
    """
    prompt = """
    Regarde cette image et classe-la dans UNE SEULE catégorie.
    Réponds UNIQUEMENT par un mot, sans ponctuation, sans phrase :

    - "satellite" : vue aérienne ou satellite d'une zone géographique
    - "sol" : photo rapprochée de sol, terre, roche, profil pédologique
    - "medical" : scanner, IRM, radiographie, échographie
    - "paysage" : photo de paysage naturel (montagne, forêt, mer…)
    - "urbain" : photo de ville, bâtiments, rues
    - "autre" : tout le reste
    """
    try:
        reponse = generer_vision(image_pil, prompt)
        reponse = reponse.strip().lower().strip('"').strip("'").strip(".")

        for mot in ["satellite", "sol", "medical", "paysage", "urbain"]:
            if mot in reponse:
                return mot
        return "autre"
    except Exception as e:
        print(f"[classifier_image] Erreur : {e}")
        return "autre"


# =====================================================
# 2.ter CLASSIFICATION DES SOLS (CNN entraîné)
# =====================================================
# =====================================================
# Dictionnaire des types de sols (fiches descriptives)
# =====================================================
FICHES_SOLS = {
    "Alluvial_Soil": {
        "nom_fr": "Sol alluvial",
        "description": "Sol déposé par les cours d'eau, riche en nutriments.",
        "couleur_typique": "Brun clair à gris",
        "texture": "Limono-sableuse à argileuse",
        "localisation": "Vallées fluviales, plaines inondables",
        "fertilité": "Élevée",
        "cultures": "Riz, maïs, blé, légumes",
    },
    "Arid_Soil": {
        "nom_fr": "Sol aride",
        "description": "Sol des régions désertiques, pauvre en matière organique.",
        "couleur_typique": "Beige à brun rouge",
        "texture": "Sableuse, caillouteuse",
        "localisation": "Déserts, zones semi-arides",
        "fertilité": "Faible (nécessite irrigation)",
        "cultures": "Dattes, agrumes (avec irrigation)",
    },
    "Black_Soil": {
        "nom_fr": "Sol noir (Vertisol)",
        "description": "Sol argileux foncé, riche en matière organique.",
        "couleur_typique": "Noir à gris foncé",
        "texture": "Argileuse lourde",
        "localisation": "Inde (Deccan), Australie, Afrique",
        "fertilité": "Très élevée",
        "cultures": "Coton, sorgho, blé",
    },
    "Laterite_Soil": {
        "nom_fr": "Sol latéritique",
        "description": "Sol rouge riche en oxydes de fer et d'aluminium.",
        "couleur_typique": "Rouge vif à brun rouge",
        "texture": "Argileuse, riche en fer",
        "localisation": "Zones tropicales humides",
        "fertilité": "Faible (carent en nutriments)",
        "cultures": "Café, thé, hévéa, anacarde",
    },
    "Mountain_Soil": {
        "nom_fr": "Sol de montagne",
        "description": "Sol mince, caillouteux, formé sur pentes.",
        "couleur_typique": "Gris à brun",
        "texture": "Caillouteuse, peu profonde",
        "localisation": "Zones montagneuses, altitudes élevées",
        "fertilité": "Variable selon l'altitude",
        "cultures": "Pommes de terre, élevage, forêts",
    },
    "Red_Soil": {
        "nom_fr": "Sol rouge",
        "description": "Sol ferralitique rouge, bien drainé.",
        "couleur_typique": "Rouge à rouge brun",
        "texture": "Argilo-sableuse",
        "localisation": "Zones tropicales et subtropicales",
        "fertilité": "Moyenne",
        "cultures": "Coton, arachide, mil",
    },
    "Yellow_Soil": {
        "nom_fr": "Sol jaune",
        "description": "Sol riche en oxydes de fer hydratés.",
        "couleur_typique": "Jaune à jaune-brun",
        "texture": "Limono-argileuse",
        "localisation": "Zones humides tempérées et tropicales",
        "fertilité": "Moyenne à élevée",
        "cultures": "Riz, soja, canne à sucre",
    },
}


def classifier_sol(image_pil):
    """
    Classification complète d'un sol :
    1. CNN pour identifier le type principal
    2. GPT-4o pour l'analyse géographique détaillée
    3. Retourne un rapport enrichi + image annotée
    """
    try:
        model, classes, transform = get_soil_classifier()

        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top3_prob, top3_idx = torch.topk(
            probabilities, min(3, len(classes))
        )

        results = []
        for i in range(len(top3_idx)):
            results.append({
                "type": classes[top3_idx[i].item()],
                "confidence": f"{top3_prob[i].item() * 100:.1f}%",
            })

        type_principal = results[0]["type"]
        confiance = results[0]["confidence"]
        fiche = FICHES_SOLS.get(type_principal, {})

        # =====================================================
        # Analyse enrichie par GPT-4o
        # =====================================================
        prompt_analyse = f"""
        Tu es un expert en pédologie, géographie et SIG.

        Une photo de sol a été analysée par un modèle de deep learning.
        Voici le résultat :
        - Type de sol identifié : {type_principal}
        - Confiance : {confiance}
        - Alternatives : {results[1:]}

        Fournis un RAPPORT D'EXPERTISE complet structuré ainsi :

        ## 1. Diagnostic principal
        Explique ce qu'est un "{type_principal}" en termes simples.
        Décris ses caractéristiques visuelles typiques.

        ## 2. Interprétation géographique
        Que nous dit cette classification sur l'environnement ?
        Climat probable, région du monde, altitude.

        ## 3. État et dynamique du sol
        - État de surface (sec, humide, fissuré, érodé…)
        - Risques identifiés (érosion, désertification, salinisation…)
        - Signes de dégradation ou de fertilité

        ## 4. Usage et recommandations
        - Cultures adaptées à ce type de sol
        - Techniques agricoles recommandées
        - Besoins en amendements ou irrigation
        - Actions de préservation

        ## 5. Analyse SIG
        - Méthodes de cartographie adaptées
        - Indices spectraux utiles (NDVI, etc.)
        - Sources de données complémentaires (Sentinel-2, Landsat)

        Sois précis et professionnel, comme un rapport d'expertise pédologique.
        """

        try:
            analyse_expert = generer_texte(
                prompt_analyse,
                system_prompt="Tu es un pédologue et expert SIG avec 20 ans d'expérience.",
            )
        except Exception as e:
            analyse_expert = f"_Analyse GPT-4o indisponible : {e}_"

        # =====================================================
        # Génération d'une image annotée simple
        # =====================================================
        import cv2
        from PIL import ImageDraw

        img_annotee = image_pil.copy()
        draw = ImageDraw.Draw(img_annotee)

        # Bandeau en bas
        w, h = img_annotee.size
        bandeau_h = 60
        bandeau = Image.new("RGB", (w, bandeau_h), (25, 25, 35))
        img_annotee.paste(bandeau, (0, h - bandeau_h))

        # Texte sur le bandeau
        texte = f"Type: {type_principal}  |  Confiance: {confiance}"
        draw.text((15, h - bandeau_h + 20), texte, fill=(255, 255, 255))

        return {
            "type_principal": type_principal,
            "confiance": confiance,
            "alternatives": results[1:],
            "fiche": fiche,
            "analyse_expert": analyse_expert,
            "image_annotee": img_annotee,
        }

    except Exception as e:
        return {"erreur": f"Erreur classification sol : {e}"}


# =====================================================
# 2.quater ANALYSE DE SOL (pipeline pédologique classique)
# =====================================================
def analyser_sol_pedologique(image_input, sujet=None):
    """
    Pipeline dédié aux photos de sol : détection fissures, végétation,
    texture, puis analyse sémantique par IA.
    Retourne (rapport_texte, image_annotee_pil, metriques_dict).
    """
    try:
        image_pil = _to_pil(image_input)
        img_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        surface_totale = img_gray.size

        # 1. Détection des fissures
        seuil = cv2.adaptiveThreshold(
            img_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2,
        )
        kernel = np.ones((3, 3), np.uint8)
        seuil = cv2.morphologyEx(seuil, cv2.MORPH_OPEN, kernel)
        masque_fissures = (seuil > 0).astype(np.uint8)
        surface_fissures = masque_fissures.sum() / surface_totale * 100

        # 2. Détection de la végétation
        masque_vert = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        surface_vegetation = (masque_vert > 0).sum() / surface_totale * 100

        # 3. Texture
        texture = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

        # 4. Couleur moyenne
        couleur_moyenne = img_np.reshape(-1, 3).mean(axis=0).astype(int)
        r, g, b = couleur_moyenne

        # 5. Création d'une image annotée
        image_annotee = img_bgr.copy()
        image_annotee[masque_fissures == 1] = [0, 0, 255]
        image_annotee[masque_vert > 0] = [0, 255, 0]

        carte_finale = cv2.addWeighted(img_bgr, 0.5, image_annotee, 0.5, 0)

        hauteur_bandeau = 50
        carte_finale = cv2.copyMakeBorder(
            carte_finale, 0, hauteur_bandeau, 0, 0,
            cv2.BORDER_CONSTANT, value=(25, 25, 25),
        )

        y_pos = carte_finale.shape[0] - 15
        cv2.rectangle(carte_finale, (15, y_pos - 10), (35, y_pos + 10), (0, 0, 255), -1)
        cv2.putText(
            carte_finale, f"Fissures: {surface_fissures:.1f}%",
            (45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.rectangle(carte_finale, (250, y_pos - 10), (270, y_pos + 10), (0, 255, 0), -1)
        cv2.putText(
            carte_finale, f"Végétation: {surface_vegetation:.1f}%",
            (280, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        carte_rgb = cv2.cvtColor(carte_finale, cv2.COLOR_BGR2RGB)
        image_annotee_pil = Image.fromarray(carte_rgb)

        metriques = {
            "surface_fissures": round(surface_fissures, 1),
            "surface_vegetation": round(surface_vegetation, 1),
            "texture": round(texture, 0),
            "couleur_moyenne_rgb": (int(r), int(g), int(b)),
        }

        prompt = f"""
        Analyse cette photo de sol à partir de mesures objectives :

        MÉTRIQUES EXTRAITES :
        - Surface fissurée : {surface_fissures:.1f}% de l'image
        - Surface végétalisée : {surface_vegetation:.1f}% de l'image
        - Indice de texture (Laplacien) : {texture:.0f}
        - Couleur moyenne RGB : ({r}, {g}, {b})

        {f"Contexte fourni par l'utilisateur : {sujet}" if sujet else ""}

        Fournis un rapport pédologique structuré :
        1. Type de sol probable (argileux, sableux, limoneux, latéritique, calcaire…)
        2. État hydrique (sec, humide, saturé) et signes de dessiccation
        3. Nature des fissures (retrait-gonflement, érosion, activité biologique)
        4. Végétation observée (densité, type probable)
        5. Recommandations agronomiques ou de gestion
        """

        try:
            rapport = generer_texte(
                prompt,
                system_prompt="Tu es un pédologue et agronome expert en analyse de sols.",
            )
        except Exception as e:
            rapport = (
                f"⚠️ L'analyse IA n'a pas pu être générée ({e}).\n\n"
                f"**Métriques objectives extraites :**\n"
                f"- Surface fissurée : {surface_fissures:.1f}%\n"
                f"- Surface végétalisée : {surface_vegetation:.1f}%\n"
                f"- Indice de texture : {texture:.0f}\n"
                f"- Couleur moyenne RGB : ({r}, {g}, {b})\n"
            )

        return rapport, image_annotee_pil, metriques

    except Exception as e:
        return f"Erreur lors de l'analyse pédologique : {e}", None, {}


# =====================================================
# 2.quinquies ANALYSE D'IMAGE GÉNÉRALE (fallback)
# =====================================================
def analyser_image_generale(image_input, sujet=None):
    """Analyse générique d'une image qui ne rentre dans aucune catégorie."""
    try:
        image_pil = _to_pil(image_input)
        prompt = f"""
        Décris et analyse cette image de manière structurée.

        {f"Contexte : {sujet}" if sujet else ""}

        Fournis :
        1. Description objective de ce que tu vois.
        2. Éléments remarquables (objets, textures, couleurs dominantes).
        3. Interprétation contextuelle (que peut-on en déduire ?).
        4. Recommandations ou points d'attention.
        """
        return generer_vision(image_pil, prompt)
    except Exception as e:
        return f"Erreur lors de l'analyse générale : {e}"


# =====================================================
# 3. UTILITAIRES
# =====================================================
def generer_palette_couleurs(nb_classes):
    """Génère des couleurs BGR distinctes."""
    couleurs_bgr = []
    for i in range(nb_classes):
        teinte = i / float(max(nb_classes, 1))
        r, g, b = colorsys.hsv_to_rgb(teinte, 0.85, 0.95)
        couleurs_bgr.append((int(b * 255), int(g * 255), int(r * 255)))
    return couleurs_bgr


def _to_pil(image_input):
    """Normalise une entrée en image PIL RGB."""
    if isinstance(image_input, str) and os.path.exists(image_input):
        img = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("Format d'image invalide.")

    if img.mode in ("RGBA", "P", "L"):
        img = img.convert("RGB")
    return img


# =====================================================
# 4. MARKETING / FINANCE / INDUSTRIE
# =====================================================
def analyser_marketing_sentiment(texte, sujet="Produit/Marque"):
    prompt = f"""
    Effectue une analyse de sentiment et d'opportunité marketing sur ce texte :

    Sujet/Marque : {sujet}
    Texte à analyser :
    \"\"\"{texte}\"\"\"

    Structure ton analyse :
    1. Sentiment Général (Positif / Neutre / Négatif + score /100).
    2. Points forts / Opportunités identifiés.
    3. Points de douleur / Critiques majeures.
    4. Intention d'achat ou d'engagement (Élevée / Moyenne / Faible).
    5. Plan d'action recommandé.
    """
    return generer_texte(
        prompt,
        system_prompt="Tu es un expert en Marketing Digital et Analyse de Sentiment.",
    )


def analyse_finance_reseau(df_transactions):
    """Analyse les liaisons et risques de fraude dans un réseau de transactions."""
    colonnes_requises = {"source", "cible", "montant"}
    manquantes = colonnes_requises - set(df_transactions.columns)
    if manquantes:
        return {
            "erreur": f"Colonnes manquantes : {', '.join(manquantes)}. "
                      f"Attendu : source, cible, montant."
        }

    G = nx.from_pandas_edgelist(
        df_transactions,
        source="source",
        target="cible",
        edge_attr="montant",
        create_using=nx.DiGraph(),
    )
    centralite = nx.betweenness_centrality(G)
    compte_cles = sorted(centralite.items(), key=lambda x: x[1], reverse=True)[:5]

    prompt = f"""
    Voici les comptes les plus centraux identifiés dans le réseau financier : {compte_cles}

    Fournis un rapport d'audit :
    1. Évaluation des risques de fraude ou de blanchiment.
    2. Identification des goulots d'étranglement ou des intermédiaires suspects.
    3. Recommandations d'audit ciblé.
    """
    analyse = generer_texte(
        prompt,
        system_prompt="Tu es un Auditeur Financier et Expert Anti-Fraude.",
    )
    return {
        "nombre_noeuds": G.number_of_nodes(),
        "nombre_liens": G.number_of_edges(),
        "rapport_audit": analyse,
    }


def analyser_vibrations_industrie(frequences, amplitudes):
    frequences_np = np.array(frequences)
    amplitudes_np = np.array(amplitudes)

    rms = float(np.sqrt(np.mean(amplitudes_np ** 2)))
    max_amp = float(np.max(amplitudes_np))
    freq_pic = float(frequences_np[np.argmax(amplitudes_np)])

    prompt = f"""
    Analyse des signaux vibratoires d'une machine industrielle :
    - Valeur efficace (RMS) : {rms:.2f} mm/s
    - Amplitude maximale : {max_amp:.2f} mm/s
    - Fréquence dominante : {freq_pic:.1f} Hz

    Fournis un diagnostic technique :
    1. Origine mécanique probable (désalignement, usure de roulements, balourd, jeu).
    2. Niveau de sévérité du risque.
    3. Recommandations d'intervention de maintenance.
    """
    diagnostic = generer_texte(
        prompt,
        system_prompt="Tu es un Ingénieur Senior en Maintenance Industrielle.",
    )
    return {
        "rms": rms,
        "amplitude_max": max_amp,
        "frequence_critique_hz": freq_pic,
        "diagnostic_maintenance": diagnostic,
    }


# =====================================================
# 5. SANTÉ & MÉDECINE
# =====================================================
def analyse_scanner_medical_vision(image_pil, prompt="Cliché Médical / Scanner"):
    """Analyse visuelle d'un examen médical."""
    prompt_complet = f"""
    Tu es un assistant virtuel spécialisé en imagerie médicale.
    Examen analysé : {prompt}

    Effectue une analyse méthodique :
    1. Description des structures anatomiques visibles et des anomalies éventuelles.
    2. Constats sur la densité, le contraste et les lésions observables.
    3. Hypothèses d'orientation diagnostique.

    Avertissement obligatoire : Document généré à titre indicatif devant être validé par un praticien de santé.
    """
    return generer_vision(image_pil, prompt_complet)


def segmenter_zones_scanner(image_pil):
    """Segmentation automatique des tissus/zones sur un cliché médical."""
    image_pil = _to_pil(image_pil)
    largeur_orig, hauteur_orig = image_pil.size
    surface_totale = largeur_orig * hauteur_orig
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    zones_identifiees = []

    try:
        model, preprocess, _ = get_segmentation_model()
        input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
        prediction = output.argmax(0).byte().cpu().numpy()
        prediction = cv2.resize(
            prediction, (largeur_orig, hauteur_orig),
            interpolation=cv2.INTER_NEAREST,
        )

        classes_uniques = np.unique(prediction)
        for class_id in classes_uniques:
            if class_id == 0:
                continue
            mask_classe = (prediction == class_id).astype(np.uint8)
            pixels_count = np.sum(mask_classe)
            pourcentage = round((pixels_count / surface_totale) * 100, 1)
            if pourcentage > 0.3:
                zones_identifiees.append({
                    "id": int(class_id),
                    "nom": f"Zone tissulaire {class_id}",
                    "pct": pourcentage,
                    "mask": mask_classe,
                })
    except Exception as e:
        print(f"Erreur lors de la segmentation IA du scanner : {e}")

    carte_annotee = (img_bgr * 0.6).astype(np.uint8)
    palette = generer_palette_couleurs(len(zones_identifiees))

    for idx, z in enumerate(zones_identifiees):
        z["couleur"] = palette[idx]
        masque_bool = (z["mask"] == 1)
        couleur_bgr = np.array(z["couleur"], dtype=np.uint8)

        zone_cible = carte_annotee[masque_bool]
        zone_couleur = np.tile(couleur_bgr, (np.sum(masque_bool), 1))

        carte_annotee[masque_bool] = cv2.addWeighted(
            zone_cible, 0.3, zone_couleur, 0.7, 0
        )

    image_rgb = cv2.cvtColor(carte_annotee, cv2.COLOR_BGR2RGB)
    pil_resultat = Image.fromarray(image_rgb)
    rapport = (
        f"Segmentation médicale terminée : "
        f"{len(zones_identifiees)} zones de densité/tissus isolées."
    )
    return pil_resultat, rapport


def analyser_statistiques_sante(df_patients):
    """Analyse statistique de données médicales."""
    stats_descriptives = (
        df_patients.describe(include="all").fillna("").to_dict()
    )
    valeurs_manquantes = df_patients.isnull().sum().to_dict()

    prompt = f"""
    Voici le résumé statistique d'une cohorte de patients / données cliniques :
    {stats_descriptives}

    Nombre de valeurs manquantes par variable :
    {valeurs_manquantes}

    Fournis une analyse statistique médicale :
    1. Résumé des moyennes, écarts-types et distributions des constantes vitales/biologiques.
    2. Identification des valeurs aberrantes ou des facteurs de risque prédominants.
    3. Recommandations pour la stratification des patients ou le suivi clinique.
    """
    analyse_llm = generer_texte(
        prompt,
        system_prompt="Tu es un Biostatisticien Médical.",
    )
    return {
        "nb_patients": len(df_patients),
        "nb_variables": len(df_patients.columns),
        "statistiques": stats_descriptives,
        "rapport_statistique_medical": analyse_llm,
    }


def diagnostic_medical_assistance(symptomes_ou_rapport):
    prompt = f"""
    Analyse le texte ou le dossier médical suivant :
    "{symptomes_ou_rapport}"

    Fournis un document synthétique structuré :
    1. Résumé clinique et antécédents marquants.
    2. Hypothèses diagnostiques principales et différentielles.
    3. Examens complémentaires conseillés (imagerie, biologie, explorations fonctionnelles).

    Avertissement : Outil d'aide à la décision clinique - ne remplace pas l'avis d'un médecin.
    """
    return generer_texte(
        prompt,
        system_prompt="Tu es un Médecin Interniste virtuel.",
    )


# =====================================================
# 6. GÉOGRAPHIE & TÉLÉDÉTECTION
# =====================================================
def analyse_image_geographique(image_input, sujet="Analyse et délimitation de zones en télédétection"):
    try:
        image_pil = _to_pil(image_input)

        prompt = f"""
        Vous êtes un expert en télédétection et SIG.
        Sujet de l'étude : {sujet}

        TÂCHE D'ANALYSE D'IMAGE SATELLITE / GÉOGRAPHIQUE :
        1. Identification des Zones : Décrivez les principales zones homogènes
           (Végétation, Urbain, Eau, Sol nu).
        2. Localisation Spatiale Relative : Précisez la position géographique
           des éléments dans l'image.
        3. Caractérisation : Texture, occupation du sol, densité de végétation.
        4. Synthèse : Enjeux d'aménagement du territoire, agricoles ou environnementaux.
        """
        return generer_vision(image_pil, prompt)
    except Exception as e:
        return f"Erreur lors de l'analyse géographique : {e}"


def delimiter_zones_teledection(image_pil):
    """Délimite géométriquement les zones sur une image satellite."""
    image_pil = _to_pil(image_pil)
    largeur_orig, hauteur_orig = image_pil.size
    surface_totale = largeur_orig * hauteur_orig
    img_np = np.array(image_pil)
    variance_couleur = float(img_np.std())
    if variance_couleur < 25:
        return image_pil, (
            "Image trop uniforme en couleur : aucune zone distincte "
            "identifiable par segmentation. Utilisez le pipeline sol/pédologie."
        )
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elements_detectes = []

    # ============================================
    # 1. Tentative DeepLabV3
    # ============================================
    try:
        model, preprocess, weights = get_segmentation_model()
        input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
        prediction = output.argmax(0).byte().cpu().numpy()
        prediction = cv2.resize(
            prediction, (largeur_orig, hauteur_orig),
            interpolation=cv2.INTER_NEAREST,
        )
        classes_uniques = np.unique(prediction)

        for class_id in classes_uniques:
            if class_id == 0:
                continue
            mask_classe = (prediction == class_id).astype(np.uint8)
            pixels_count = np.sum(mask_classe)
            pourcentage = round((pixels_count / surface_totale) * 100, 1)
            if pourcentage > 0.5:
                nom_trouve = weights.meta["categories"][class_id]
                elements_detectes.append({
                    "id": int(class_id),
                    "nom": nom_trouve.capitalize(),
                    "pct": pourcentage,
                    "mask": mask_classe,
                })
    except Exception as e:
        print(f"Erreur segmentation DeepLab: {e}")

    # ============================================
    # 2. Fallback K-Means amélioré
    # ============================================
    if not elements_detectes:
        try:
            k = 5
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            data_pixels = img_hsv.reshape((-1, 3)).astype(np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                20, 0.5,
            )
            _, labels, _ = cv2.kmeans(
                data_pixels, k, None, criteria, 20,
                cv2.KMEANS_PP_CENTERS,
            )
            labels = labels.reshape((hauteur_orig, largeur_orig))

            for idx in range(k):
                mask_k = (labels == idx).astype(np.uint8)
                kernel = np.ones((7, 7), np.uint8)
                mask_k = cv2.morphologyEx(mask_k, cv2.MORPH_OPEN, kernel)
                mask_k = cv2.morphologyEx(mask_k, cv2.MORPH_CLOSE, kernel)

                pct = round((np.sum(mask_k) / surface_totale) * 100, 1)
                if pct > 1.0:
                    elements_detectes.append({
                        "id": idx + 1,
                        "nom": f"Zone {idx + 1}",
                        "pct": pct,
                        "mask": mask_k,
                    })
        except Exception as e:
            print(f"Erreur segmentation K-Means: {e}")

    # ============================================
    # 3. Identification sémantique par GPT-4o
    # ============================================
    labels_zones = {}
    if elements_detectes:
        try:
            description_zones = "Zones détectées par segmentation :\n"
            for idx, el in enumerate(elements_detectes):
                mask = el["mask"].astype(bool)
                if np.any(mask):
                    couleur_bgr = img_bgr[mask].mean(axis=0).astype(int)
                    couleur_rgb = (
                        int(couleur_bgr[2]),
                        int(couleur_bgr[1]),
                        int(couleur_bgr[0]),
                    )
                    description_zones += (
                        f"- Zone {idx+1} : Couleur RGB ≈ {couleur_rgb}, "
                        f"couvre {el['pct']}% de l'image.\n"
                    )

            prompt_llm = f"""
            Tu es un expert en télédétection. Une image satellite a été segmentée
            en plusieurs zones. Chaque zone est décrite par sa couleur moyenne (RGB)
            et son pourcentage de surface.

            {description_zones}

            En te basant UNIQUEMENT sur ces informations (couleur + pourcentage),
            donne pour chaque zone un nom d'occupation du sol parmi :

            - Eau (bleu profond, souvent en bas de l'image)
            - Végétation dense (vert foncé, forte proportion)
            - Végétation clairsemée (vert clair, moyen)
            - Sol nu (marron, beige, sable)
            - Zone urbaine (gris, blanc, structures)
            - Roche (gris foncé, texture)
            - Cultures (vert clair régulier, parcelles)
            - Ombre / Nuages (noir ou blanc uniforme)

            Réponds EXACTEMENT sous forme d'objet JSON, sans texte autour :
            {{"Zone 1": "Végétation dense", "Zone 2": "Sol nu", "Zone 3": "Eau"}}
            """

            reponse_llm = generer_texte(
                prompt_llm,
                system_prompt=(
                    "Tu es un expert en télédétection. "
                    "Tu réponds UNIQUEMENT par un objet JSON valide."
                ),
            )

            import re
            match = re.search(r'\{.*\}', reponse_llm, re.DOTALL)
            if match:
                labels_zones = json.loads(match.group())
                print(f"[delimiter_zones] Labels identifiés : {labels_zones}")
        except Exception as e:
            print(f"[delimiter_zones] Erreur identification LLM : {e}")

    # ============================================
    # 4. Dessin des zones
    # ============================================
    nb_elements = len(elements_detectes)
    palette = generer_palette_couleurs(nb_elements)
    carte_sig = (img_bgr * 0.55).astype(np.uint8)

    for idx, el in enumerate(elements_detectes):
        el["couleur"] = palette[idx]
        masque_bool = (el["mask"] == 1)
        couleur_bgr = np.array(el["couleur"], dtype=np.uint8)

        carte_sig[masque_bool] = cv2.addWeighted(
            carte_sig[masque_bool], 0.3,
            np.tile(couleur_bgr, (np.sum(masque_bool), 1)), 0.7, 0,
        )

        contours, _ = cv2.findContours(
            el["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            aire = cv2.contourArea(c)
            if aire > (surface_totale * 0.01):
                epsilon = 0.01 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                cv2.drawContours(carte_sig, [approx], -1, (255, 255, 255), 2)

    # ============================================
    # 5. Bandeau de légende
    # ============================================
    hauteur_legende = 60
    image_finale = cv2.copyMakeBorder(
        carte_sig, 0, hauteur_legende, 0, 0,
        cv2.BORDER_CONSTANT, value=(25, 25, 25),
    )

    if nb_elements > 0:
        largeur_colonne = int(largeur_orig / nb_elements)
        y_pos = hauteur_orig + 35

        for idx, el in enumerate(elements_detectes):
            x_pos = 15 + (idx * largeur_colonne)
            nom_zone = labels_zones.get(f"Zone {idx+1}", el["nom"])

            cv2.rectangle(
                image_finale, (x_pos, y_pos - 12),
                (x_pos + 18, y_pos + 8), el["couleur"], -1,
            )
            cv2.rectangle(
                image_finale, (x_pos, y_pos - 12),
                (x_pos + 18, y_pos + 8), (255, 255, 255), 1,
            )
            texte = f"{nom_zone}: {el['pct']}%"
            cv2.putText(
                image_finale, texte, (x_pos + 24, y_pos + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )

    image_rgb = cv2.cvtColor(image_finale, cv2.COLOR_BGR2RGB)
    pil_result = Image.fromarray(image_rgb)
    description_sig = (
        f"Délimitation effectuée : {nb_elements} zones spatiales identifiées."
    )

    return pil_result, description_sig


def lire_geotiff(chemin_ou_fichier):
    """Lecture et conversion d'un GeoTIFF."""
    if rasterio is None:
        raise ImportError("La bibliothèque 'rasterio' n'est pas installée.")

    with rasterio.open(chemin_ou_fichier) as src:
        if src.count >= 3:
            data = src.read([1, 2, 3])
            img_array = reshape_as_image(data)
        else:
            data = src.read(1)
            img_array = np.stack([data] * 3, axis=-1)

        if img_array.dtype != np.uint8:
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (
                    (img_array - img_min) / (img_max - img_min) * 255
                ).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)

        meta = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "bounds": src.bounds,
        }

        image_pil = Image.fromarray(img_array)
        return image_pil, meta


# =====================================================
# 7. STATISTIQUES GÉNÉRALES / POWER BI
# =====================================================
def analyse_dataframe(df, sujet=None):
    dimensions = (
        f"Dimensions du jeu de données : "
        f"{len(df)} lignes et {len(df.columns)} colonnes.\n"
    )
    colonnes_info = df.dtypes.astype(str).to_dict()
    valeurs_manquantes = df.isnull().sum().to_dict()

    stats_numeriques = (
        df.describe().to_dict()
        if not df.select_dtypes(include=[np.number]).empty
        else "Aucune variable numérique"
    )

    prompt = f"""
    Effectue une analyse décisionnelle approfondie de ce jeu de données :

    1. Structure : {dimensions}
    2. Colonnes et Types : {colonnes_info}
    3. Valeurs manquantes : {valeurs_manquantes}
    4. Résumé statistique numérique : {stats_numeriques}

    {f"Sujet d'étude spécifique : {sujet}" if sujet else ""}

    Livre un rapport d'analyse structuré :
    - Synthèse globale et qualité des données.
    - Principales tendances, corrélations et constats clés.
    - Recommandations d'actions opérationnelles / business.
    """
    return generer_texte(
        prompt,
        system_prompt="Tu es un Data Analyst et Consultant Decisionnel Senior.",
    )


def generer_table_powerbi(df, n_rows=10):
    return df.head(n_rows).to_html(
        classes="table table-striped table-hover table-bordered table-stats",
        index=False,
        border=0,
    )


def get_plot_data_json(df):
    """
    Exporte les données au format JSON pour les graphiques Plotly.
    Détecte et convertit les colonnes numériques (même en string).
    """
    if df is None or df.empty:
        return "{}"

    df_work = df.copy()

    for col in df_work.columns:
        if df_work[col].dtype == object:
            cleaned = (
                df_work[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() >= 0.7 * len(converted):
                df_work[col] = converted

    num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        return "{}"

    df_export = df_work.head(1000).copy()
    df_export = df_export.where(pd.notna(df_export), None)

    payload = {
        "columns": df_export.columns.tolist(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "records": df_export.to_dict(orient="records"),
    }

    return json.dumps(payload, ensure_ascii=False, default=str)