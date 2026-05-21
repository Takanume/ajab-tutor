"""Feynman-Robinson RAG — Streamlit app v1 (bilingue EN + FR, défaut EN).

Architecture mode-switch :
  1. Classifier LLM range la requête en CHEAT / UNDERSTAND / MEANING.
     Décision A1 : si le classifier renvoie autre chose, default → UNDERSTAND.
  2. Retrieval Chroma top-K=4 sur la partition (mode, langue).
     Décision A2 : si retrieve renvoie [], court-circuit → message "reformule".
  3. Generation LLM avec system prompt mode+langue + chunks + historique multi-tour.
     Décision Q1 : wrappé en try/except global, message ami si fail.

UI :
  - Sélecteur de langue dans la sidebar (radio English/Français, défaut English)
  - Chat input + historique
  - Caption "Mode: X" visible sous chaque réponse (décision A1)
  - 3 boutons sous chaque réponse pour rerun la question dans un autre mode

Run :  streamlit run app.py
"""
from __future__ import annotations

import os

# IMPORTANT: doit être set AVANT l'import chromadb pour contourner le bug protobuf
# descriptor sur Python 3.14 + opentelemetry-proto pre-generated files. Force protobuf
# à utiliser le parser pure-Python (plus tolérant, marginalement plus lent).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import APIConnectionError, APITimeoutError, Groq, RateLimitError

import prompts

load_dotenv()

# === Configuration ===

CHROMA_DIR = Path(__file__).parent / "data" / "chroma"

CLASSIFIER_MODEL = "llama-3.3-70b-versatile"  # 8B-instant ratait les CHEAT — 70B passe baseline
GENERATION_MODEL = "llama-3.3-70b-versatile"

VALID_MODES = ("CHEAT", "UNDERSTAND", "MEANING")
VALID_LANGS = ("en", "fr")
DEFAULT_LANG = "en"

# (mode, lang) → nom de collection Chroma (cf. ingest.py)
def collection_name_for(mode: str, lang: str) -> str:
    base = {"CHEAT": "feynman_refuse", "UNDERSTAND": "feynman_explain", "MEANING": "robinson_meaning"}[mode]
    return f"{base}_{lang}"


MODE_TO_EMOJI = {"CHEAT": "🛑", "UNDERSTAND": "💡", "MEANING": "🌱"}

# Multi-turn memory : nombre de tours user/assistant gardés en contexte.
HISTORY_TURNS_KEPT = 5


# === i18n ===

LOCALES = {
    "en": {
        "page_title": "AJAB Tutor",
        "app_title": "📚 AJAB Tutor",
        "app_caption": (
            "Ask me your question. I'll detect what you're looking for and reply in the matching style."
        ),
        "welcome_hero": """
<div style="padding: 24px 26px; background: linear-gradient(135deg, rgba(239, 68, 68, 0.10) 0%, rgba(249, 115, 22, 0.06) 100%); border-left: 6px solid #ef4444; border-radius: 14px; margin: 8px 0 22px 0;">
  <h2 style="margin: 0; font-size: 1.75em; font-weight: 800; line-height: 1.25; letter-spacing: -0.01em;">
    You don't hate learning.<br>
    You hate learning things that <span style="color: #ef4444;">make no sense</span>.
  </h2>
</div>

<p style="font-size: 1.08em; margin: 14px 0 8px 0; opacity: 0.78;">
ChatGPT gives you the answer. <em>You'll forget it by Friday.</em>
</p>

<div style="margin: 10px 0 22px 0; padding: 14px 18px; background: rgba(249, 115, 22, 0.12); border-left: 5px solid #f97316; border-radius: 10px;">
  <span style="font-size: 1.15em; font-weight: 700; color: #ea580c;">I refuse to give you the answer.</span>
  <span style="font-size: 1.05em;"> And that's exactly why you'll remember it.</span>
</div>

<p style="font-size: 1.02em; margin: 16px 0; line-height: 1.55;">
Complex subjects, broken down step by step, until they click. The kind of understanding that opens doors
<span style="background: rgba(139, 92, 246, 0.18); padding: 3px 10px; border-radius: 8px; font-weight: 700; color: #7c3aed; white-space: nowrap;">next year</span>
in university interviews,
<span style="background: rgba(139, 92, 246, 0.18); padding: 3px 10px; border-radius: 8px; font-weight: 700; color: #7c3aed; white-space: nowrap;">in three years</span>
during your first internship — not just "someday".
</p>

<div style="margin: 26px 0 14px 0; padding: 18px 22px; background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); color: white; border-radius: 14px; text-align: center; box-shadow: 0 4px 14px rgba(139, 92, 246, 0.28);">
  <div style="font-size: 1.18em; font-weight: 700; margin-bottom: 4px;">👇 Ask your first question below.</div>
  <div style="font-size: 0.95em; opacity: 0.93; font-weight: 400;">Even one you'd be embarrassed to ask in class.</div>
</div>
""",
        "sidebar_about": "How I work",
        "sidebar_modes_intro": "Three modes, picked automatically based on what you actually need:",
        "sidebar_mode_understand": "- 💡 **Understand** — you don't get a concept and want it to click",
        "sidebar_mode_meaning": "- 🌱 **Meaning** — you want to know why this matters in real life",
        "sidebar_mode_cheat": "- 🛑 **No shortcut** — you want the answer; I refuse, and walk you through it instead",
        "sidebar_modes_outro": (
            "I read what you write to pick the mode. Wrong guess? Tap the buttons under any answer to switch."
        ),
        "sidebar_lang_label": "Language",
        "sidebar_lang_help": "Switch language. Resets the conversation.",
        "sidebar_reset_button": "🔄 New conversation",
        "label_understand": "UNDERSTAND",
        "label_meaning": "MEANING",
        "label_cheat": "NO SHORTCUT",
        "mode_caption": "Mode",
        "mode_forced_caption": "Mode (forced)",
        "spinner_thinking": "Thinking...",
        "spinner_force_mode": "Answering in {mode} mode...",
        "input_placeholder": "Ask a question (concept you don't get, meaning, homework...)",
        "friendly_error": "The tutor is thinking... Try again in 5 seconds.",
        "rate_limit_error": (
            "⏳ The tutor is busy right now — too many people are asking at the same time. "
            "Please wait about 30 seconds and try again. "
            "(Free tier limit: ~30 questions/minute shared across all users of this app.)"
        ),
        "network_error": (
            "🌐 Connection issue with the AI service. Check your internet, then try again in a moment."
        ),
        "empty_retrieval": (
            "I don't have the exact context to answer that. Could you rephrase with other words, "
            "or be more specific about the concept that's blocking you?"
        ),
        "err_no_groq_key": (
            "GROQ_API_KEY not found. Create one at https://console.groq.com then paste it into `.env` "
            "(see .env.example)."
        ),
        "err_no_chroma": "Chroma store not found at {path}. Run `python ingest.py` once before the first `streamlit run`.",
    },
    "fr": {
        "page_title": "AJAB Tutor",
        "app_title": "📚 AJAB Tutor",
        "app_caption": (
            "Pose-moi ta question. Je détecte ce que tu cherches et je te réponds dans le style adapté."
        ),
        "welcome_hero": """
<div style="padding: 24px 26px; background: linear-gradient(135deg, rgba(239, 68, 68, 0.10) 0%, rgba(249, 115, 22, 0.06) 100%); border-left: 6px solid #ef4444; border-radius: 14px; margin: 8px 0 22px 0;">
  <h2 style="margin: 0; font-size: 1.75em; font-weight: 800; line-height: 1.25; letter-spacing: -0.01em;">
    Tu détestes pas apprendre.<br>
    Tu détestes apprendre <span style="color: #ef4444;">pour rien</span>.
  </h2>
</div>

<p style="font-size: 1.08em; margin: 14px 0 8px 0; opacity: 0.78;">
ChatGPT te donne la réponse. <em>Tu l'oublies vendredi.</em>
</p>

<div style="margin: 10px 0 22px 0; padding: 14px 18px; background: rgba(249, 115, 22, 0.12); border-left: 5px solid #f97316; border-radius: 10px;">
  <span style="font-size: 1.15em; font-weight: 700; color: #ea580c;">Moi je refuse de te la donner.</span>
  <span style="font-size: 1.05em;"> Et c'est pour ça que tu vas vraiment retenir.</span>
</div>

<p style="font-size: 1.02em; margin: 16px 0; line-height: 1.55;">
On décortique le sujet ensemble, étape par étape, jusqu'au déclic. Pas pour le contrôle de demain — pour
<span style="background: rgba(139, 92, 246, 0.18); padding: 3px 10px; border-radius: 8px; font-weight: 700; color: #7c3aed; white-space: nowrap;">l'oral du bac</span>,
<span style="background: rgba(139, 92, 246, 0.18); padding: 3px 10px; border-radius: 8px; font-weight: 700; color: #7c3aed; white-space: nowrap;">Parcoursup</span>,
<span style="background: rgba(139, 92, 246, 0.18); padding: 3px 10px; border-radius: 8px; font-weight: 700; color: #7c3aed; white-space: nowrap;">ton premier stage</span>.
</p>

<div style="margin: 26px 0 14px 0; padding: 18px 22px; background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); color: white; border-radius: 14px; text-align: center; box-shadow: 0 4px 14px rgba(139, 92, 246, 0.28);">
  <div style="font-size: 1.18em; font-weight: 700; margin-bottom: 4px;">👇 Tape ta première question en bas.</div>
  <div style="font-size: 0.95em; opacity: 0.93; font-weight: 400;">Même celle que tu n'oserais jamais poser en classe.</div>
</div>
""",
        "sidebar_about": "Comment je marche",
        "sidebar_modes_intro": "Trois modes, choisis automatiquement selon ce que tu cherches vraiment :",
        "sidebar_mode_understand": "- 💡 **Comprendre** — tu piges pas un concept et tu veux que ça fasse déclic",
        "sidebar_mode_meaning": "- 🌱 **Sens** — tu veux savoir pourquoi ça compte dans la vraie vie",
        "sidebar_mode_cheat": "- 🛑 **Pas de raccourci** — tu veux la réponse ; je refuse, et je te fais avancer toi-même",
        "sidebar_modes_outro": (
            "Je lis ce que tu écris pour choisir le mode. Mauvaise pioche ? Clique les boutons sous n'importe quelle réponse pour switcher."
        ),
        "sidebar_lang_label": "Langue",
        "sidebar_lang_help": "Change la langue. Réinitialise la conversation.",
        "sidebar_reset_button": "🔄 Nouvelle conversation",
        "label_understand": "COMPRENDRE",
        "label_meaning": "SENS",
        "label_cheat": "PAS DE RACCOURCI",
        "mode_caption": "Mode",
        "mode_forced_caption": "Mode (forcé)",
        "spinner_thinking": "Je réfléchis...",
        "spinner_force_mode": "Je te réponds en mode {mode}...",
        "input_placeholder": "Pose ta question (concept que tu comprends pas, sens, devoir...)",
        "friendly_error": "Le tuteur réfléchit... Réessaie dans 5 secondes.",
        "rate_limit_error": (
            "⏳ Le tuteur est très demandé en ce moment — trop de questions arrivent en même temps. "
            "Attends environ 30 secondes et réessaie. "
            "(Limite free tier : ~30 questions/minute partagées entre tous les utilisateurs de l'app.)"
        ),
        "network_error": (
            "🌐 Problème de connexion au service d'IA. Vérifie ton réseau et réessaie dans un instant."
        ),
        "empty_retrieval": (
            "Je n'ai pas le contexte exact pour te répondre. Tu peux reformuler avec d'autres mots, "
            "ou être plus précis sur le concept qui te bloque ?"
        ),
        "err_no_groq_key": (
            "GROQ_API_KEY non trouvée. Crée une clé sur https://console.groq.com puis colle-la dans `.env` "
            "(voir .env.example)."
        ),
        "err_no_chroma": "Chroma store introuvable à {path}. Lance `python ingest.py` une fois avant le premier `streamlit run`.",
    },
}

LANG_DISPLAY_NAMES = {"en": "English", "fr": "Français"}


def t(key: str, lang: str, **fmt) -> str:
    """Lookup i18n. Fallback sur EN si la clé manque dans la langue demandée."""
    text = LOCALES.get(lang, LOCALES[DEFAULT_LANG]).get(key) or LOCALES[DEFAULT_LANG].get(key, key)
    return text.format(**fmt) if fmt else text


def mode_label(mode: str, lang: str) -> str:
    """Label localisé d'un mode."""
    return t({"CHEAT": "label_cheat", "UNDERSTAND": "label_understand", "MEANING": "label_meaning"}[mode], lang)


def format_error_message(exc: Exception, lang: str) -> str:
    """Map une exception Groq au message localisé approprié.

    - RateLimitError (HTTP 429) : message dédié rate-limit avec contexte free-tier.
    - APIConnectionError / APITimeoutError : message dédié réseau.
    - Autre : friendly_error générique (fallback Q1).
    """
    if isinstance(exc, RateLimitError):
        return t("rate_limit_error", lang)
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return t("network_error", lang)
    return t("friendly_error", lang)


# === Pipeline ===


@st.cache_resource
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Affiché dans la langue active si déjà set dans session_state, sinon défaut
        lang = st.session_state.get("lang", DEFAULT_LANG)
        st.error(t("err_no_groq_key", lang))
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource
def get_chroma_client() -> chromadb.api.ClientAPI:
    # Auto-bootstrap : si data/chroma/ n'existe pas (premier boot d'un fresh deploy
    # type Streamlit Cloud où data/chroma/ est gitignored), lance l'ingest une fois.
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        import ingest
        with st.spinner("First-time setup: building the knowledge base (~30s)..."):
            try:
                ingest.main()
            except Exception as exc:
                lang = st.session_state.get("lang", DEFAULT_LANG)
                st.error(f"Ingest failed: {exc}")
                st.stop()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def classify_intent(client: Groq, question: str, lang: str) -> str:
    """Classifier la requête. A1 : fallback UNDERSTAND si réponse off-list."""
    classifier_prompt = prompts.CLASSIFIER_PROMPT_BY_LANG[lang]
    response = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        max_tokens=8,
        temperature=0.0,
        messages=[
            {"role": "system", "content": classifier_prompt},
            {"role": "user", "content": question},
        ],
    )
    raw = (response.choices[0].message.content or "").strip().upper()
    for valid in VALID_MODES:
        if valid in raw:
            return valid
    return "UNDERSTAND"  # décision A1


def retrieve_chunks(chroma_client: chromadb.api.ClientAPI, mode: str, lang: str, question: str, k: int = 4) -> list[str]:
    """Top-K chunks de la partition (mode, langue). Retourne [] si collection vide ou absente."""
    coll_name = collection_name_for(mode, lang)
    try:
        collection = chroma_client.get_collection(coll_name)
    except Exception:
        return []
    try:
        results = collection.query(query_texts=[question], n_results=k)
    except Exception:
        return []
    docs = results.get("documents", [[]])
    if not docs or not docs[0]:
        return []
    return list(docs[0])


def generate_response(
    client: Groq,
    mode: str,
    lang: str,
    chunks: list[str],
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Génération avec system prompt (mode, langue) + chunks RAG + historique multi-tour."""
    instructions = prompts.MODE_PROMPT_BY_LANG[(mode, lang)]
    context_header = "\n\n=== EXTRAITS / EXCERPTS ===\n"
    context_block = context_header + "\n\n---\n\n".join(chunks) + "\n=== END ==="
    messages: list[dict] = [{"role": "system", "content": instructions + context_block}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=700,
        temperature=0.7,
        messages=messages,
    )
    return response.choices[0].message.content or t("friendly_error", lang)


def answer(
    client: Groq,
    chroma_client: chromadb.api.ClientAPI,
    question: str,
    lang: str,
    forced_mode: str | None = None,
    history: list[dict] | None = None,
) -> tuple[str, str]:
    """Pipeline complet. Retourne (response_text, mode_used).

    Décisions :
      - A1 : forced_mode None → classifier ; off-list → UNDERSTAND.
      - A2 : retrieve == [] → court-circuit avec EMPTY_RETRIEVAL_MSG.
      - Q1 : le caller wrappe en try/except.
      - Multi-turn : history dans la génération uniquement.
      - i18n : lang propagée à classify, retrieve, generate (3 niveaux du pipeline).
    """
    if forced_mode in VALID_MODES:
        mode = forced_mode
    else:
        mode = classify_intent(client, question, lang)
    chunks = retrieve_chunks(chroma_client, mode, lang, question)
    if not chunks:
        return t("empty_retrieval", lang), mode
    text = generate_response(client, mode, lang, chunks, question, history=history)
    return text, mode


def build_history(messages: list[dict], turns_to_keep: int = HISTORY_TURNS_KEPT) -> list[dict]:
    """Convertit st.session_state.messages en format OpenAI et cap aux N derniers tours."""
    pruned: list[dict] = [{"role": m["role"], "content": m["content"]} for m in messages]
    max_messages = turns_to_keep * 2
    if len(pruned) > max_messages:
        pruned = pruned[-max_messages:]
    return pruned


# === UI ===


def render_assistant_buttons(question: str, current_mode: str, msg_index: int, lang: str) -> str | None:
    """3 boutons sous une réponse assistant pour rerun la question dans un autre mode."""
    other_modes = [m for m in VALID_MODES if m != current_mode]
    cols = st.columns(len(other_modes))
    for col, mode in zip(cols, other_modes):
        with col:
            if st.button(
                f"{MODE_TO_EMOJI[mode]} {mode_label(mode, lang)}",
                key=f"force_{mode}_{msg_index}",
                use_container_width=True,
            ):
                return mode
    return None


def render_sidebar(lang: str) -> str:
    """Affiche la sidebar et retourne la langue éventuellement modifiée."""
    with st.sidebar:
        # Sélecteur de langue
        lang_options = list(VALID_LANGS)
        new_lang = st.radio(
            t("sidebar_lang_label", lang),
            options=lang_options,
            index=lang_options.index(lang),
            format_func=lambda code: LANG_DISPLAY_NAMES[code],
            help=t("sidebar_lang_help", lang),
            key="lang_selector",
        )
        st.divider()
        st.header(t("sidebar_about", new_lang))
        st.markdown(
            t("sidebar_modes_intro", new_lang)
            + "\n"
            + t("sidebar_mode_understand", new_lang)
            + "\n"
            + t("sidebar_mode_meaning", new_lang)
            + "\n"
            + t("sidebar_mode_cheat", new_lang)
            + "\n\n"
            + t("sidebar_modes_outro", new_lang)
        )
        st.divider()
        if st.button(t("sidebar_reset_button", new_lang)):
            st.session_state.messages = []
            st.rerun()
    return new_lang


def main() -> None:
    # Init session state
    if "lang" not in st.session_state:
        st.session_state.lang = DEFAULT_LANG
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_rerun" not in st.session_state:
        st.session_state.pending_rerun = None

    lang = st.session_state.lang

    st.set_page_config(
        page_title=t("page_title", lang),
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Injecte le styling custom du chat input (cohérence avec la hero violet/rose).
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title(t("app_title", lang))

    new_lang = render_sidebar(lang)
    if new_lang != lang:
        # Changement de langue → reset historique pour éviter de garder du français dans une conv anglaise et inversement
        st.session_state.lang = new_lang
        st.session_state.messages = []
        st.rerun()

    # Welcome hero : affiché uniquement à l'ouverture (état vide), disparaît dès la première
    # question. Sert de "landing pitch" qui ne pollue pas la conversation longue session.
    # unsafe_allow_html=True nécessaire pour le styling inline (gradients, pills, callout panels).
    if not st.session_state.messages and not st.session_state.pending_rerun:
        st.markdown(t("welcome_hero", lang), unsafe_allow_html=True)
    else:
        st.caption(t("app_caption", lang))

    # Render history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("mode"):
                st.caption(f"{MODE_TO_EMOJI[msg['mode']]} {t('mode_caption', lang)} : {mode_label(msg['mode'], lang)}")
                if i == len(st.session_state.messages) - 1:
                    forced = render_assistant_buttons(msg["question"], msg["mode"], i, lang)
                    if forced:
                        st.session_state.pending_rerun = (msg["question"], forced)
                        st.rerun()

    # Handle pending rerun (force mode)
    if st.session_state.pending_rerun:
        question, forced_mode = st.session_state.pending_rerun
        st.session_state.pending_rerun = None
        history_for_rerun = build_history(st.session_state.messages[:-2]) if len(st.session_state.messages) >= 2 else []
        with st.chat_message("assistant"):
            with st.spinner(t("spinner_force_mode", lang, mode=mode_label(forced_mode, lang))):
                try:
                    response_text, mode_used = answer(
                        get_groq_client(),
                        get_chroma_client(),
                        question,
                        lang,
                        forced_mode=forced_mode,
                        history=history_for_rerun,
                    )
                except Exception as exc:  # Q1 — classifie par type pour message ciblé
                    response_text, mode_used = format_error_message(exc, lang), forced_mode
            st.markdown(response_text)
            st.caption(f"{MODE_TO_EMOJI[mode_used]} {t('mode_forced_caption', lang)} : {mode_label(mode_used, lang)}")
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "mode": mode_used, "question": question}
        )
        st.rerun()

    # Input
    user_input = st.chat_input(t("input_placeholder", lang))
    if user_input:
        history = build_history(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner(t("spinner_thinking", lang)):
                try:
                    response_text, mode_used = answer(
                        get_groq_client(),
                        get_chroma_client(),
                        user_input,
                        lang,
                        history=history,
                    )
                except Exception as exc:  # Q1 — classifie par type pour message ciblé
                    response_text = format_error_message(exc, lang)
                    mode_used = "UNDERSTAND"
            st.markdown(response_text)
            st.caption(f"{MODE_TO_EMOJI[mode_used]} {t('mode_caption', lang)} : {mode_label(mode_used, lang)}")
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "mode": mode_used, "question": user_input}
        )
        st.rerun()


if __name__ == "__main__":
    main()
