# AJAB Tutor — anti-ChatGPT tutor for high school students

A RAG-based pedagogical tutor that adapts its answer style to what the student is actually trying to do. The thesis: massive ChatGPT use among high school students is a symptom; root causes are either "I don't understand" or "I don't see the point". The system detects the intent of each request and replies in the matching style.

**Bilingual (EN + FR), default English.** Switch language via the sidebar radio.

---

## Français

Un tuteur RAG qui adapte son style de réponse à ce que l'élève essaie réellement de faire. La thèse : l'usage massif de ChatGPT chez les lycéens est un symptôme dont les causes-racines sont soit "je ne comprends pas" soit "je ne vois pas le sens". Le système détecte l'intent de la requête et répond selon le mode adapté.

Bilingue (EN + FR), défaut anglais. Bascule la langue via le radio dans la sidebar.

POC privé v1, non commercial. Voir `~/.gstack/projects/Takanume-l3jeb/` pour le design doc complet.

## Architecture

```
User question
     │
     ▼
[Classifier]  → CHEAT / UNDERSTAND / MEANING  (default UNDERSTAND si doute)
     │
     ▼
[Retrieval]   → Chroma top-K=4 sur la partition correspondant au mode
     │
     ▼  (si retrieve vide → court-circuit + "reformule")
[Generation] → system prompt mode-spécifique + chunks retrieved
     │            wrappé en try/except (message ami si fail)
     ▼
[Streamlit UI] → réponse + indicateur "Mode: X" + bouton "changer mode"
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# édite .env et colle ta clé Groq (créer sur https://console.groq.com — free tier)
```

LLM utilisé : Llama 3.1 8B Instant (classifier) + Llama 3.3 70B Versatile (génération), hébergés sur Groq free tier (~30 req/min, largement assez pour les démos).

## Pipeline d'ingestion (1 fois, ~5 min)

```bash
python ingest.py
```

Télécharge et chunke les passages Feynman (Caltech public) et Robinson (TED transcripts), tague par persona, embed dans Chroma local (`data/chroma/`).

## Lancer l'app

```bash
streamlit run app.py
```

Ouvre `http://localhost:8501`.

## Eval du classifier

```bash
python eval.py
```

Boucle sur `evals/classifier.json` (15-20 cas) et affiche le pass/fail count. Baseline minimum pour shipper la v1 : ≥ 80 %.

## Décisions de design

| ID | Sujet | Choix |
|----|-------|-------|
| A1 | Fallback classifier | default-to-UNDERSTAND + bouton "changer mode" visible |
| A2 | Empty retrieval | hard fail + message "Je n'ai pas le contexte exact, reformule ?" |
| Q1 | Error handling démo | try/except global + message ami |
| T1 | Eval classifier | JSON cases + script eval.py reproductible |

## Limitations connues (v1)

- Conversation multi-tour limitée à 5 derniers tours (`HISTORY_TURNS_KEPT` dans `app.py:55`).
- Corpus paraphrasé pour POC privé. Pour usage commercial, remplacer par contenu sous licence.
- Corpus EN minimal (3 fichiers, 1 par mode) vs FR (6 fichiers). Ajouter des `.en.txt` dans `corpus/` quand tu veux étoffer.
- Embeddings ChromaDB par défaut (anglais-first, MiniLM-L6-v2). FR fonctionne mais retrieval suboptimal. Upgrade multilangue en v1.5.
- Pas de deploy public. Local Streamlit only.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public or private — both work on free tier).
2. Go to https://share.streamlit.io and sign in with your GitHub.
3. **New app** → pick this repo → branch `main` → main file `app.py`.
4. **Advanced settings → Secrets** → paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. **Deploy**. First boot takes ~30 sec (downloads the embedding model, runs `ingest.py` automatically). Subsequent boots are fast.

You'll get a URL like `https://<username>-ajab-tutor-app.streamlit.app`. Share it directly with the people you want to demo to.

## Adding a new corpus file

Drop a `.txt` file in `corpus/feynman/` or `corpus/robinson/` with 3 headers:

```
# tag: feynman-explain    # one of: feynman-explain, feynman-refuse, robinson-meaning
# lang: en                # or: fr
# source: <attribution>

<your text here, plain prose, paragraphs separated by blank lines>
```

Then re-run `python ingest.py` to repopulate Chroma.

## Prochaine étape

Faire 6 conversations avec 3 parents + 3 lycéens AVANT d'itérer. Le wedge "anti-ChatGPT compound debt" reste non validé tant que personne ne l'a essayé en vrai.
