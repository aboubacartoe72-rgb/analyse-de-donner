import base64
import io
import json
import traceback

import markdown

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.safestring import mark_safe

# Imports des modules locaux
from .data_loader import load_file
from .nlp_engine import analyse_text
from .visualise import (
    generate_interactive_plot,
    generate_summary_plots,
)
from .advanced_engine import (
    analyser_marketing_sentiment,
    analyse_finance_reseau,
    analyser_vibrations_industrie,
    analyse_scanner_medical_vision,
    segmenter_zones_scanner,
    analyser_statistiques_sante,
    diagnostic_medical_assistance,
    analyse_image_geographique,
    delimiter_zones_teledection,
    lire_geotiff,
    analyse_dataframe,
    generer_table_powerbi,
    get_plot_data_json,
    # ✅ Fonctions de classification
    classifier_image,
    classifier_sol,
    analyser_sol_pedologique,
    analyser_image_generale,
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DOMAINES_VALIDES = [
    "MARKETING & FINANCE & INDUSTRIE & SANTE",
    "SANTE & MEDECINE",
    "GEOGRAPHIE & TELEDETECTION",
    "STATISTIQUES GÉNÉRALES & POWER BI",
]

EXTENSIONS_TEXTE = {".pdf", ".txt", ".docx", ".md"}


# ---------------------------------------------------------------------------
# Utilitaire de formatage HTML des stats
# ---------------------------------------------------------------------------
def format_stats_to_html(stats_dict):
    """Génère une structure HTML propre et moderne pour les statistiques."""
    if not stats_dict or not isinstance(stats_dict, dict):
        return "<p>Aucune statistique disponible.</p>"

    html = []
    html.append('<div class="kpi-conteneur">')

    for key, label in [
        ("total_lignes", "Total Lignes"),
        ("total_colonnes", "Total Colonnes"),
        ("valeurs_manquantes", "Valeurs Manquantes"),
    ]:
        if key in stats_dict:
            html.append(f'''
                <div class="kpi-card">
                    <div class="kpi-title">{label}</div>
                    <div class="kpi-value">{stats_dict[key]}</div>
                </div>
            ''')

    html.append('</div>')

    stats_num = stats_dict.get('stats_numeriques', {})
    if stats_num:
        html.append('<h3>Statistiques Descriptives</h3>')
        html.append('<table class="table-stats">')
        html.append('''
            <thead>
                <tr>
                    <th>Colonne</th>
                    <th class="text-end">Moyenne</th>
                    <th class="text-end">Min</th>
                    <th class="text-end">Max</th>
                    <th class="text-end">Somme</th>
                    <th class="text-end">Écart-type</th>
                    <th class="text-end">Médiane</th>
                    <th class="text-end">CV (%)</th>
                    <th class="text-end">Quartiles</th>
                </tr>
            </thead>
            <tbody>
        ''')

        if isinstance(stats_num, dict):
            items_to_iterate = stats_num.items()
        elif isinstance(stats_num, list):
            items_to_iterate = [
                (item.get('colonne', f'Colonne {i+1}'), item)
                if isinstance(item, dict) else (f'Colonne {i+1}', {})
                for i, item in enumerate(stats_num)
            ]
        else:
            items_to_iterate = []

        def fmt(values, key):
            v = values.get(key)
            return f"{v:,.2f}" if isinstance(v, (int, float)) else (v or "-")

        for col, values in items_to_iterate:
            if not isinstance(values, dict):
                continue

            html.append(f'''
                <tr>
                    <td><strong>{col}</strong></td>
                    <td class="text-end">{fmt(values, 'moyenne')}</td>
                    <td class="text-end">{fmt(values, 'min')}</td>
                    <td class="text-end">{fmt(values, 'max')}</td>
                    <td class="text-end">{fmt(values, 'somme')}</td>
                    <td class="text-end">{fmt(values, 'ecart_type')}</td>
                    <td class="text-end">{fmt(values, 'mediane')}</td>
                    <td class="text-end">{fmt(values, 'Coefficient_de_Variation')}</td>
                    <td class="text-end">{fmt(values, 'quartiles')}</td>
                </tr>
            ''')

        html.append('</tbody></table>')

    return "".join(html)


# ---------------------------------------------------------------------------
# Vue d'accueil
# ---------------------------------------------------------------------------
def index(request):
    return render(request, 'index.html')


# ---------------------------------------------------------------------------
# API : domaine sélectionné
# ---------------------------------------------------------------------------
@csrf_exempt
def api_domaine_selected(request):
    if request.method != 'POST':
        return JsonResponse(
            {"status": "error", "message": "Méthode non autorisée"},
            status=405
        )

    domaine = request.POST.get('domaine', '').strip()

    if domaine not in DOMAINES_VALIDES:
        return JsonResponse(
            {"status": "error", "message": "Domaine invalide"},
            status=400
        )

    request.session['domaine'] = domaine

    return JsonResponse({
        "status": "success",
        "domaine": domaine,
        "message": f"Domaine '{domaine}' sélectionné",
    })


# ---------------------------------------------------------------------------
# API : analyse multi-domaine
# ---------------------------------------------------------------------------
@csrf_exempt
def api_analyse_multidomaine(request):
    if request.method != 'POST':
        return JsonResponse(
            {"status": "error", "message": "Méthode non autorisée"},
            status=405
        )

    domaine = request.POST.get('domaine')

    try:
        if domaine == 'marketing':
            texte = request.POST.get('texte', '')
            res = analyser_marketing_sentiment(texte)
            return JsonResponse({
                "status": "success", "domaine": "marketing", "resultat": res
            })

        elif domaine == 'sante':
            rapport = request.POST.get('texte', '')
            res = diagnostic_medical_assistance(rapport)
            return JsonResponse({
                "status": "success", "domaine": "sante", "resultat": res
            })

        elif domaine == 'industrie':
            freq = json.loads(request.POST.get('frequences', '[]'))
            amp = json.loads(request.POST.get('amplitudes', '[]'))
            res = analyser_vibrations_industrie(freq, amp)
            return JsonResponse({
                "status": "success", "domaine": "industrie", "resultat": res
            })

        return JsonResponse(
            {"status": "error", "message": "Domaine non reconnu"},
            status=400
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {"status": "error", "message": "JSON invalide pour fréquences/amplitudes"},
            status=400
        )
    except Exception as e:
        traceback.print_exc()
        return JsonResponse(
            {"status": "error", "message": f"Erreur serveur : {str(e)}"},
            status=500
        )


# ---------------------------------------------------------------------------
# Vue : analyse texte simple
# ---------------------------------------------------------------------------
def analyser_view(request):
    if request.method == "POST":
        sujet = request.POST.get("sujet", "").strip()
        if not sujet:
            return render(request, "index.html", {
                "erreur": "Veuillez saisir un sujet."
            })

        resultat = analyser_marketing_sentiment(sujet)
        return render(request, "index.html", {
            "resultats": mark_safe(markdown.markdown(resultat))
        })

    return render(request, "index.html")


# ---------------------------------------------------------------------------
# Vue principale : analyse de fichier
# ---------------------------------------------------------------------------
def analyse_fichier(request):
    """Traite le fichier, le domaine et le sujet envoyés en POST."""

    if request.method != 'POST':
        return render(request, 'index.html')

    if 'file' not in request.FILES:
        return render(request, 'index.html', {
            'erreur': 'Aucun fichier fourni.'
        })

    uploaded_file = request.FILES['file']
    domaine = request.POST.get('domaine', '').strip()
    sujet = request.POST.get('sujet', '').strip()

    context = {
        "fichier": uploaded_file.name,
        "domaine": domaine,
        "sujet": sujet,
        "resultats": mark_safe("<p>Aucun résultat n'a été généré.</p>"),
        "graphique": None,
        "graphique_interactif": None,
        "table_powerbi": None,
        "plot_data_json": None,
        "is_image": False,
        "image_delimitee": None,
        "rapport_teledection": None,
        "type_image_detecte": None,
    }

    try:
        data_type, data = load_file(uploaded_file)

        resultats = None
        graphique_base64 = None
        graphique_interactif = None
        table_html = None
        plot_json = None
        is_image = False
        image_delimitee_data = None
        rapport_teledection = None
        type_image_detecte = None

        # ==================================================================
        # CAS 1 : DataFrame
        # ==================================================================
        if data_type == "dataframe":
            resultats = (
                analyse_dataframe(data, sujet=sujet)
                if sujet else analyse_dataframe(data)
            )
            graphique_base64 = generate_summary_plots(data)
            graphique_interactif = generate_interactive_plot(data)

            table_html = generer_table_powerbi(data)
            if table_html and isinstance(table_html, str):
                table_html = table_html.replace(
                    'class="dataframe"', 'class="table-stats"'
                )
                if 'class="table-stats"' not in table_html:
                    table_html = table_html.replace(
                        '<table>', '<table class="table-stats">'
                    )

            plot_json = get_plot_data_json(data)
            if not plot_json or plot_json == "{}":
                print("[analyse_fichier] Aucune donnée numérique — pas de graphique.")

        # ==================================================================
        # CAS 2 : Texte
        # ==================================================================
        elif data_type == "texte":
            if domaine == "SANTE & MEDECINE":
                resultats = diagnostic_medical_assistance(data)
            elif domaine == "MARKETING & FINANCE & INDUSTRIE & SANTE":
                resultats = analyser_marketing_sentiment(data)
            else:
                resultats = (
                    analyse_text(data, sujet=sujet)
                    if 'analyse_text' in globals() else str(data)
                )

        # ==================================================================
        # CAS 3 : Image — ROUTAGE INTELLIGENT
        # ==================================================================
        elif data_type == "image":
            is_image = True

            # ---- 3.1 : Classification automatique ----
            type_image_detecte = classifier_image(data)
            print(f"[analyse_fichier] Type d'image détecté : {type_image_detecte}")

            # ---- 3.2 : Routage ----
            if type_image_detecte == "medical":
                resultats = analyse_scanner_medical_vision(data, prompt=sujet)

            elif type_image_detecte == "sol":
                # ✅ Classification CNN + analyse expert GPT-4o
                resultat_cnn = classifier_sol(data)
                print(
                    f"[analyse_fichier] Résultat CNN : "
                    f"{resultat_cnn.get('type_principal', 'N/A')}"
                )

                if "erreur" in resultat_cnn:
                    # Fallback : pipeline pédologique classique
                    rapport_sol, image_annotee_sol, metriques_sol = (
                        analyser_sol_pedologique(data, sujet=sujet)
                    )
                    resultats = rapport_sol
                    rapport_teledection = metriques_sol

                    if image_annotee_sol is not None:
                        try:
                            if image_annotee_sol.mode != 'RGB':
                                image_annotee_sol = image_annotee_sol.convert('RGB')
                            buffer = io.BytesIO()
                            image_annotee_sol.save(buffer, format="JPEG")
                            img_str = base64.b64encode(
                                buffer.getvalue()
                            ).decode('utf-8')
                            image_delimitee_data = (
                                f"data:image/jpeg;base64,{img_str}"
                            )
                        except Exception as e:
                            print(f"Erreur encodage image sol : {e}")
                else:
                    # ✅ Rapport enrichi CNN + expert
                    type_principal = resultat_cnn.get("type_principal", "Inconnu")
                    confiance = resultat_cnn.get("confiance", "N/A")
                    analyse_expert = resultat_cnn.get("analyse_expert", "")
                    alts = resultat_cnn.get("alternatives", [])

                    alts_txt = "\n".join(
                        [f"- {a['type']} : {a['confidence']}" for a in alts]
                    ) if alts else "_Aucune alternative_"

                    resultats = f"""
## 🔬 Classification du sol (CNN)

**Type principal : {type_principal}**
**Confiance : {confiance}**

### Autres possibilités
{alts_txt}

---

{analyse_expert}
"""

                    # Encoder l'image annotée en base64
                    image_annotee = resultat_cnn.get("image_annotee")
                    if image_annotee is not None:
                        try:
                            if image_annotee.mode != 'RGB':
                                image_annotee = image_annotee.convert('RGB')
                            buffer = io.BytesIO()
                            image_annotee.save(buffer, format="JPEG")
                            img_str = base64.b64encode(
                                buffer.getvalue()
                            ).decode('utf-8')
                            image_delimitee_data = (
                                f"data:image/jpeg;base64,{img_str}"
                            )
                        except Exception as e:
                            print(f"Erreur encodage image annotée : {e}")

                    rapport_teledection = None

            elif type_image_detecte == "satellite":
                consigne_systeme = (
                    "Analyse cette image satellite d'un point de vue "
                    "géographique et télédétection. Identifie l'occupation "
                    "du sol, le relief, les zones homogènes."
                )
                prompt_final = (
                    f"{consigne_systeme}\n\nSujet spécifique : {sujet}"
                    if sujet else consigne_systeme
                )
                resultats = analyse_image_geographique(data, sujet=prompt_final)

                try:
                    res_teledection = delimiter_zones_teledection(data)

                    if isinstance(res_teledection, (tuple, list)):
                        image_annotee = res_teledection[0]
                        if len(res_teledection) > 1:
                            rapport_teledection = res_teledection[1]
                    else:
                        image_annotee = res_teledection

                    if image_annotee and hasattr(image_annotee, 'convert'):
                        if image_annotee.mode != 'RGB':
                            image_annotee = image_annotee.convert('RGB')

                        buffer = io.BytesIO()
                        image_annotee.save(buffer, format="JPEG")
                        img_str = base64.b64encode(
                            buffer.getvalue()
                        ).decode('utf-8')
                        image_delimitee_data = (
                            f"data:image/jpeg;base64,{img_str}"
                        )

                except Exception as img_err:
                    print(f"Erreur lors de la délimitation : {img_err}")

            elif type_image_detecte == "paysage":
                resultats = analyser_image_generale(data, sujet=sujet)

            elif type_image_detecte == "urbain":
                resultats = analyser_image_generale(data, sujet=sujet)

            else:  # "autre" ou fallback
                if domaine == "SANTE & MEDECINE":
                    resultats = analyse_scanner_medical_vision(data, prompt=sujet)
                else:
                    resultats = analyser_image_generale(data, sujet=sujet)

        # ==================================================================
        # Conversion du résultat en HTML
        # ==================================================================
        if resultats:
            if isinstance(resultats, dict):
                if "stats_numeriques" in resultats or "total_lignes" in resultats:
                    resultats_html = format_stats_to_html(resultats)
                else:
                    texte_markdown = (
                        resultats.get('analyse')
                        or resultats.get('texte')
                        or resultats.get('resultat')
                        or str(resultats)
                    )
                    resultats_html = markdown.markdown(
                        texte_markdown,
                        extensions=['tables', 'fenced_code', 'nl2br']
                    )
            else:
                resultats_html = markdown.markdown(
                    str(resultats),
                    extensions=['tables', 'fenced_code', 'nl2br']
                )
        else:
            resultats_html = "<p>Aucun résultat n'a été généré.</p>"

        context.update({
            "resultats": mark_safe(resultats_html),
            "graphique": graphique_base64,
            "graphique_interactif": (
                mark_safe(graphique_interactif) if graphique_interactif else None
            ),
            "table_powerbi": mark_safe(table_html) if table_html else None,
            "plot_data_json": (
                plot_json if plot_json and plot_json != "{}" else None
            ),
            "is_image": is_image,
            "image_delimitee": image_delimitee_data,
            "rapport_teledection": rapport_teledection,
            "type_image_detecte": type_image_detecte,
        })

        return render(request, 'resultats.html', context)

    except Exception as e:
        traceback.print_exc()
        return render(request, 'index.html', {
            'erreur': f"Erreur lors de l'analyse : {str(e)}"
        })