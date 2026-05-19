"""Eval reproductible du classifier d'intent — bilingue (EN + FR) (décision T1).

Lit evals/classifier.json, lance classify_intent() sur chaque cas dans sa langue
respective, affiche pass/fail global ET par langue.
Baseline minimum visée pour shipper la v1 : >= 80% pass rate (global).

Run :  python eval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from app import classify_intent

load_dotenv()

EVAL_FILE = Path(__file__).parent / "evals" / "classifier.json"
BASELINE = 0.80


def main() -> int:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("error: GROQ_API_KEY non trouvée (voir .env.example)", file=sys.stderr)
        return 1
    if not EVAL_FILE.exists():
        print(f"error: {EVAL_FILE} introuvable", file=sys.stderr)
        return 1

    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    client = Groq(api_key=api_key)

    passes = 0
    fails: list[dict] = []
    per_lang_total: dict[str, int] = defaultdict(int)
    per_lang_pass: dict[str, int] = defaultdict(int)

    print(f"Eval classifier sur {len(cases)} cas (multi-langue)...\n")
    for i, case in enumerate(cases, 1):
        question = case["question"]
        expected = case["expected_intent"]
        lang = case.get("lang", "fr")  # fallback FR pour rétro-compat
        per_lang_total[lang] += 1
        try:
            actual = classify_intent(client, question, lang)
        except Exception as exc:
            print(f"  [{i:2d}] [{lang}] ERROR  '{question[:50]}...' → {exc}")
            fails.append({**case, "actual": f"ERROR: {exc}"})
            continue
        ok = actual == expected
        mark = "PASS" if ok else "FAIL"
        snippet = question[:55] + ("..." if len(question) > 55 else "")
        print(f"  [{i:2d}] [{lang}] {mark}  expected={expected:11s}  actual={actual:11s}  '{snippet}'")
        if ok:
            passes += 1
            per_lang_pass[lang] += 1
        else:
            fails.append({**case, "actual": actual})

    rate = passes / len(cases) if cases else 0
    print(f"\nRésultat global : {passes}/{len(cases)} pass = {rate:.0%}")
    print(f"Baseline minimum : {BASELINE:.0%}")
    print("Détail par langue :")
    for lang in sorted(per_lang_total):
        total = per_lang_total[lang]
        ok = per_lang_pass[lang]
        lang_rate = ok / total if total else 0
        print(f"  {lang}: {ok}/{total} = {lang_rate:.0%}")

    if fails:
        print(f"\n{len(fails)} cas échoués :")
        for f in fails:
            print(f"  - [{f.get('lang', '??')}] '{f['question']}'")
            print(f"      expected={f['expected_intent']}  actual={f['actual']}")
            if "note" in f:
                print(f"      note: {f['note']}")

    if rate < BASELINE:
        print(f"\nverdict: SOUS BASELINE ({rate:.0%} < {BASELINE:.0%}) — tune le prompt avant ship.")
        return 1
    print("\nverdict: OK pour shipper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
