"""
evaluate_responses.py
---------------------
Evaluates model accuracy by checking the 'conclusion' column of:
  - response_gemma_2.csv
  - response_queen_2.csv

Ground truth: ALL images show objects/animals in CORRECT relative sizes.
Therefore:
  - Conclusion starts with "Yes"  → CORRECT prediction (True Positive)
  - Conclusion starts with "No"   → INCORRECT prediction (False Negative)
  - Anything else                 → UNCERTAIN / unparseable
"""

import csv
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────


# Size/correctness keywords
_SIZE_KW = re.compile(
    r"\b(correct|accurate|proper|right|appropriate|proportional|relative size)\b",
    re.IGNORECASE,
)
# Broad negative: 'not' appearing anywhere within 120 chars before a size keyword
_NEGATIVE_BROAD = re.compile(
    r"\bnot\b.{0,120}\b(correct|accurate|proper|right|appropriate|proportional)\b",
    re.IGNORECASE | re.DOTALL,
)


def classify_conclusion(text: str) -> str:
    """
    Return 'yes' or 'no' (no more 'uncertain') based on the conclusion text.

    Strategy:
      1. 'Yes'/'No' prefix  → definitive.
      2. If 'not' appears within 120 chars before a correctness keyword → 'no'.
      3. If a correctness keyword appears at all (without a preceding 'not') → 'yes'.
      4. Last resort: treat the response as incorrect (conservative).
    """
    if not text or not isinstance(text, str):
        return "no"   # treat missing/empty as incorrect
    stripped = text.strip()
    lower = stripped.lower()

    # ── Step 1: prefix check ──────────────────────────────────────────────
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"

    # ── Step 2: broad negative pattern ───────────────────────────────────
    if _NEGATIVE_BROAD.search(stripped):
        return "no"

    # ── Step 3: any positive size keyword present (no 'not' before it) ───
    if _SIZE_KW.search(stripped):
        return "yes"

    # ── Step 4: last resort — treat as incorrect ──────────────────────────
    print(f"Uncertain: {text}")
    return "no"


def parse_csv(filepath: str) -> list[dict]:
    """
    Robustly parse a CSV that may have quoted multi-line fields.
    Returns a list of row dicts.
    """
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def evaluate(rows: list[dict], model_name: str) -> dict:
    """
    Evaluate model accuracy.
    Ground truth: every image is CORRECTLY sized → expected answer = 'yes'.
    """
    total = len(rows)
    counts = Counter()
    wrong_examples = []

    for row in rows:
        conclusion = row.get("conclusion", "")
        label = classify_conclusion(conclusion)
        counts[label] += 1

        #verdict = row.get("verdict", "")
        #label = classify_conclusion(verdict)
        #counts[label] += 1

        if label == "no":
            wrong_examples.append({
                "file": row.get("file_name", "?"),
                "conclusion": conclusion.strip()[:120],
            })


    correct   = counts["yes"]
    incorrect = counts["no"]
    accuracy  = correct / total * 100 if total > 0 else 0.0

    return {
        "model":         model_name,
        "total":         total,
        "correct":       correct,
        "incorrect":     incorrect,
        "accuracy_pct":  accuracy,
        "wrong_examples": wrong_examples,
    }


def print_report(result: dict, show_examples: int = 5) -> None:
    """Pretty-print evaluation results."""
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  Model : {result['model']}")
    print(f"{'═' * 60}")
    print(f"  Total responses  : {result['total']}")
    print(f"  ✅ Correct  (Yes) : {result['correct']}")
    print(f"  ❌ Incorrect (No) : {result['incorrect']}")
    print(sep)
    print(f"  🎯 Accuracy       : {result['accuracy_pct']:.2f}%")
    print(f"{'═' * 60}")

    if result["wrong_examples"] and show_examples > 0:
        print(f"\n  ── Sample wrong predictions (first {show_examples}) ──")
        for ex in result["wrong_examples"][:show_examples]:
            print(f"    [{ex['file']}] {ex['conclusion']}")


# ── plotting ─────────────────────────────────────────────────────────────────


def plot_results(results: list[dict], save_path: str = "evaluation_comparison.png") -> None:
    """
    Create a grouped bar chart comparing model evaluation results.
    Shows Correct / Incorrect counts per model with accuracy annotation.
    """
    models    = [r["model"] for r in results]
    correct   = [r["correct"] for r in results]
    incorrect = [r["incorrect"] for r in results]
    accuracy  = [r["accuracy_pct"] for r in results]
    totals    = [r["total"] for r in results]

    x = np.arange(len(models))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_correct   = ax.bar(x - bar_width / 2, correct,   bar_width,
                            label="Correct (Yes)", color="#2ecc71", edgecolor="white")
    bars_incorrect = ax.bar(x + bar_width / 2, incorrect, bar_width,
                            label="Incorrect (No)", color="#e74c3c", edgecolor="white")

    # annotate counts on bars
    for bar in bars_correct:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, str(int(h)),
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    for bar in bars_incorrect:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, str(int(h)),
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    # annotate accuracy below model name
    for i, (acc, tot) in enumerate(zip(accuracy, totals)):
        ax.text(x[i], -max(totals) * 0.07,
                f"{acc:.1f}% acc\n(n={tot})",
                ha="center", va="top", fontsize=9, color="#555555")

    ax.set_ylabel("Number of Responses", fontsize=12)
    ax.set_title("Model Evaluation: Correct vs Incorrect Predictions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(correct), max(incorrect)) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n  📊 Bar chart saved to: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

FILES = {
    "Gemma":  "response_gemma_2.csv",
    "Qwen":   "response_qwen_2.csv",
}

if __name__ == "__main__":
    results = []

    for model_name, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        print(f"Parsing {filepath} …", end=" ", flush=True)
        rows = parse_csv(filepath)
        print(f"{len(rows)} rows loaded.")

        result = evaluate(rows, model_name)
        results.append(result)
        print_report(result)

    # ── Side-by-side summary ─────────────────────────────────────────────────
    if len(results) >= 2:
        print(f"\n{'═' * 60}")
        print("  COMPARISON SUMMARY")
        print(f"{'═' * 60}")
        header = f"  {'Model':<12} {'Total':>6} {'Correct':>8} {'Incorrect':>10} {'Accuracy':>10}"
        print(header)
        print(f"  {'─'*50}")
        for r in results:
            print(f"  {r['model']:<12} {r['total']:>6} {r['correct']:>8} {r['incorrect']:>10} {r['accuracy_pct']:>9.2f}%")
        print(f"{'═' * 60}\n")

        # Winner
        accs = [(r["model"], r["accuracy_pct"]) for r in results]
        best = max(accs, key=lambda x: x[1])
        print(f"  🏆 Best model: {best[0]}  ({best[1]:.2f}% accuracy)\n")

    # ── Bar chart ────────────────────────────────────────────────────────────
    if results:
        plot_results(results)
