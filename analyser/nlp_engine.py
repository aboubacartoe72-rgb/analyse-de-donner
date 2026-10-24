import ollama


def analyse_text(text, sujet=None):
    """
    Analyse un texte brut avec Ollama (modèle local, gratuit, illimité).
    """
    # Construit le prompt en tenant compte du sujet si fourni
    contexte = f"\nContexte / sujet de l'analyse : {sujet}\n" if sujet else ""

    prompt = f"""
    Tu es un assistant d'entreprise expert. Lis attentivement le document ci-dessous.
    {contexte}
    Identifie toutes les questions posées et réponds-y de manière claire et détaillée
    en t'appuyant sur les données du texte.

    Document :
    {text}
    """

    try:
        # Appel du modèle local Ollama (gratuit, aucune clé API)
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[
                {
                    'role': 'system',
                    'content': "Tu es un assistant d'analyse documentaire."
                },
                {'role': 'user', 'content': prompt}
            ]
        )
        reponse_ia = response['message']['content']

    except Exception as e:
        reponse_ia = (
            f"⚠️ Erreur avec Ollama. Vérifiez que l'application Ollama "
            f"tourne en arrière-plan et que le modèle 'llama3.2:1b' "
            f"est téléchargé (commande : ollama pull llama3.2:1b).\n\n"
            f"Détail technique : {str(e)}"
        )

    return {
        "nb_caracteres": len(text),
        "reponses_questions": reponse_ia
    }