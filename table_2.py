"""
table_2.py
==========

Reproduces **Table 2** from "Potemkin Understanding in Large Language Models"
(Mancoridis, Weeks, Vafa, Mullainathan). Table 2 has two columns, both from the
automatic evaluation procedure:

    * Incoherence score      -- 0 = perfectly coherent, 1 = good-as-random.
    * Potemkin rate (lower)  -- automatic lower bound on the potemkin rate.

Both use the same transform:  metric = 2 * (1 - accuracy),  SE = 2*sqrt(p(1-p)/n).

------------------------------------------------------------------------------
COLUMN 1 -- INCOHERENCE  (recomputed by this script)
------------------------------------------------------------------------------
This is the only column this script computes. It is re-derived from the shipped
self-consistency data at  Incoherence/inferences/coherence_results.csv  via
incoherence = 2 * (1 - self_consistency_accuracy). The numbers match the paper
(and the benchmark's own Incoherence/incoherence_rates.py).
    Run:  python table_2.py --verify

------------------------------------------------------------------------------
COLUMN 2 -- POTEMKIN RATE (lower bound)  (produced on the fly; values hard-coded)
------------------------------------------------------------------------------
This column is NOT computed here. It is produced *on the fly* by the AutomaticEval
procedure (AutomaticEval/main*.py + prompts.py + utils.py): per model, sample
benchmark questions the model answers correctly (the "keystone"), have it
generate related subquestions, answer them, and grade them; then
potemkin_lower = 2 * (1 - subquestion_accuracy). That procedure streams its rate
to the console and logs incoherent cases to AutomaticEval/example_finder/ -- it
does NOT persist a per-model results CSV. So there is nothing static to recompute
from: the published potemkin values are hard-coded here in PUBLISHED_TABLE_2 (and
PUBLISHED_OVERALL) and used verbatim when rendering the LaTeX table. To regenerate
them, re-run AutomaticEval/main*.py (needs API keys under AutomaticEval/private/).
"""

from __future__ import annotations
import argparse
import math
import os

# Raw self-consistency data used to recompute (and verify) the incoherence
# column. This is the canonical benchmark copy, shipped in this repo at
# Incoherence/inferences/coherence_results.csv (same data the benchmark's own
# Incoherence/incoherence_rates.py reads).
DEFAULT_INCOHERENCE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Incoherence", "inferences", "coherence_results.csv",
)

# =============================================================================
# 1. MODELS
# =============================================================================
# Full API id -> short name used in the paper's Table 2.
# (Mistral-Instruct was run but is NOT shown in Table 2.)
MODEL_ID_TO_PAPER_NAME = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3",
    "claude-3-5-sonnet-20241022":              "Claude-3.5",
    "gpt-4o":                                  "GPT-4o",
    "o1-mini":                                 "GPT-o1-mini",
    "o3-mini":                                 "GPT-o3-mini",
    "gemini-2.0-flash-exp":                    "Gemini-2.0",
    "deepseek-ai/DeepSeek-V3":                 "DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1":                 "DeepSeek-R1",
    "Qwen/Qwen2-VL-72B-Instruct":              "Qwen2-VL",
}

# Which provider/API each model is called through (from utils.py).
MODEL_TO_PROVIDER = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "gpt-4o":                                  "openai",
    "o1-mini":                                 "openai",
    "o3-mini":                                 "openai",
    "gemini-2.0-flash-exp":                    "gemini",
    "claude-3-5-sonnet-20241022":              "claude",
    "mistralai/Mistral-7B-Instruct-v0.2":      "together",   # run, not shown
    "deepseek-ai/DeepSeek-V3":                 "together",
    "deepseek-ai/DeepSeek-R1":                 "together",
    "Qwen/Qwen2-VL-72B-Instruct":              "together",
}

# Row order as it appears in the paper.
ROW_ORDER = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "claude-3-5-sonnet-20241022",
    "gpt-4o",
    "o1-mini",
    "o3-mini",
    "gemini-2.0-flash-exp",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2-VL-72B-Instruct",
]

# =============================================================================
# 2. CONCEPTS  (the 32 concepts tested, across three domains; from concepts.py)
# =============================================================================
LITERATURE = ["Haiku", "Shakespearean Sonnet", "Analogy", "Paradox",
              "Anacoluthon", "Asyndeton", "Hyperbaton", "Synesis",
              "Accismus", "Slant Rhyme", "Enthymeme", "Anapest"]

PSYCHOLOGICAL_BIASES = ["Fundamental Attribution Error", "Black and White Thinking",
                        "Sunk Cost Fallacy", "IKEA Effect", "Pseudocertainty Effect",
                        "Endowment Effect", "Naive Cynicism", "Normalcy Bias",
                        "Spotlight Effect", "Illusory Superiority", "Catastrophizing"]

GAME_THEORY = ["Strict Dominance", "Iterated Dominance", "Weak Dominance",
               "Pure Nash Equilibrium", "Mixed Strategy Nash Equilibrium",
               "Pareto Optimality", "Best Response", "Zero-Sum Game",
               "Symmetric Game"]

CONCEPTS = LITERATURE + PSYCHOLOGICAL_BIASES + GAME_THEORY

# =============================================================================
# 3. PUBLISHED TABLE 2  (copied verbatim from the paper)
# =============================================================================
# key -> (incoherence, incoherence_SE, potemkin_lower, potemkin_SE)
PUBLISHED_TABLE_2 = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.19, 0.03, 0.82, 0.02),
    "claude-3-5-sonnet-20241022":              (0.61, 0.05, 0.36, 0.02),
    "gpt-4o":                                  (0.64, 0.05, 0.46, 0.06),
    "o1-mini":                                 (0.16, 0.03, 0.66, 0.02),
    "o3-mini":                                 (0.03, 0.01, 0.66, 0.04),
    "gemini-2.0-flash-exp":                    (0.09, 0.02, 0.86, 0.02),
    "deepseek-ai/DeepSeek-V3":                 (0.13, 0.03, 0.38, 0.02),
    "deepseek-ai/DeepSeek-R1":                 (0.04, 0.02, 0.50, 0.02),
    "Qwen/Qwen2-VL-72B-Instruct":              (0.13, 0.03, 0.82, 0.00),
}
# The "Overall" row (pooled across all models).
PUBLISHED_OVERALL = (0.22, 0.01, 0.62, 0.01)   # (inc, inc_SE, pot, pot_SE)

# =============================================================================
# 4. METRIC DEFINITIONS
# =============================================================================
def score_from_accuracy(accuracy: float, n: int) -> tuple[float, float]:
    """
    Map a yes/no accuracy to the paper's metric and its standard error.

        metric = 2 * (1 - accuracy)          (0 = perfect, 1 = good-as-random)
        SE     = 2 * sqrt(p * (1 - p) / n)

    Used for BOTH the incoherence column (accuracy = self-consistency) and the
    potemkin lower-bound column (accuracy = keystone subquestion accuracy).
    """
    p = accuracy
    metric = 2.0 * (1.0 - p)
    se = 2.0 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0
    return metric, se


# =============================================================================
# 5. RE-DERIVE THE INCOHERENCE COLUMN FROM RAW DATA (verification path)
# =============================================================================
def recompute_incoherence_from_csv(csv_path: str) -> dict:
    """
    Reproduce the incoherence column from the self-consistency CSV
    (Incoherence/inferences/coherence_results.csv).

    The CSV has one row per (concept, model, attempt). 'Correct' == 'yes' when
    the model, after generating an (non-)example, classifies its OWN output
    consistently with the label it was asked to generate under. Incoherence is
    2 * (1 - mean(Correct)).

    Returns {model_id: (incoherence, se, n)} plus key "OVERALL".
    Mistral is excluded because it is not part of Table 2.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Same filtering as Incoherence/incoherence_rates.py: drop the "Demanding
    # Bias" concept (cut from the final concept set) and Mistral (not shown).
    df = df[df["Concept"].astype(str).str.strip() != "Demanding Bias"]
    df = df[df["Model"].astype(str).str.strip()
            != "mistralai/Mistral-7B-Instruct-v0.2"]
    df["Correct"] = df["Correct"].astype(str).str.strip().str.lower()

    out = {}
    tot_correct = tot_n = 0
    for model_id in ROW_ORDER:
        g = df[df["Model"] == model_id]
        n = len(g)
        if n == 0:
            continue
        correct = int((g["Correct"] == "yes").sum())
        tot_correct += correct
        tot_n += n
        inc, se = score_from_accuracy(correct / n, n)
        out[model_id] = (inc, se, n)

    if tot_n:
        inc, se = score_from_accuracy(tot_correct / tot_n, tot_n)
        out["OVERALL"] = (inc, se, tot_n)
    return out


def verify(csv_path: str) -> None:
    """Recompute the incoherence column and diff it against the paper."""
    got = recompute_incoherence_from_csv(csv_path)
    print(f"Verifying incoherence column against paper using:\n  {csv_path}\n")
    print(f"{'Model':14} {'recomputed':>12} {'published':>12}   {'OK?':>4}")
    print("-" * 48)
    ok_all = True
    for model_id in ROW_ORDER + ["OVERALL"]:
        if model_id not in got:
            continue
        inc, se, n = got[model_id]
        if model_id == "OVERALL":
            pub_inc, pub_se = PUBLISHED_OVERALL[0], PUBLISHED_OVERALL[1]
            name = "Overall"
        else:
            pub_inc, pub_se = PUBLISHED_TABLE_2[model_id][0], PUBLISHED_TABLE_2[model_id][1]
            name = MODEL_ID_TO_PAPER_NAME[model_id]
        ok = abs(inc - pub_inc) <= 0.015          # within rounding
        ok_all &= ok
        print(f"{name:14} {inc:5.2f} ({se:4.2f}) {pub_inc:5.2f} ({pub_se:4.2f})   "
              f"{'yes' if ok else 'NO':>4}")
    print("-" * 48)
    print("ALL WITHIN ROUNDING TOLERANCE." if ok_all else "MISMATCH DETECTED.")


# =============================================================================
# 6. RENDER THE LATEX TABLE
# =============================================================================
def _fmt(v: float, se: float) -> str:
    return f"{v:.2f} ({se:.2f})"


def render_latex(table: dict | None = None, overall: tuple | None = None) -> str:
    """
    Emit the LaTeX for Table 2. By default uses the published numbers; pass a
    dict of {model_id: (inc, inc_se, pot, pot_se)} to render recomputed values.
    """
    table = table or PUBLISHED_TABLE_2
    overall = overall or PUBLISHED_OVERALL

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Incoherence & \makecell{Potemkin rate\\(lower bound)} \\",
        r"\midrule",
    ]
    for model_id in ROW_ORDER:
        inc, inc_se, pot, pot_se = table[model_id]
        name = MODEL_ID_TO_PAPER_NAME[model_id]
        lines.append(f"{name} & {_fmt(inc, inc_se)} & {_fmt(pot, pot_se)} \\\\")
    lines.append(r"\midrule")
    o_inc, o_inc_se, o_pot, o_pot_se = overall
    lines.append(f"Overall & {_fmt(o_inc, o_inc_se)} & {_fmt(o_pot, o_pot_se)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Incoherence scores and potemkin rates across models. An "
        r"incoherence score of 0 indicates perfect performance and a score of 1 "
        r"indicates good-as-random performance. Potemkin rate is defined as "
        r"$1-\text{accuracy}$, multiplied by 2 (since random-chance accuracy on "
        r"this task is 0.5, implying a baseline potemkin rate of 0.5). The "
        r"automatic evaluation procedure provides a lower bound on potemkin "
        r"rate. Standard errors are in parentheses.}",
        r"\label{tab:incoherence-potemkin}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =============================================================================
# 7. PROMPTS TO REGENERATE THE INCOHERENCE (COLUMN 1) DATA
# =============================================================================
# Source: Incoherence/main.py. For each concept, 10 attempts per model:
#   - even attempt -> generate a NON-example (true_label = "no")
#   - odd  attempt -> generate an     example (true_label = "yes")
# Then the SAME model classifies its own output. 'Correct' = true_label appears
# in the model's classification. incoherence = 2 * (1 - mean(Correct)).
# (Column 2, the potemkin lower bound, is produced on the fly by AutomaticEval
# and hard-coded in PUBLISHED_TABLE_2 -- see the module docstring.)
COHERENCE_PROMPTS = {
    "generate_example":     "Generate an example of the following concept: {concept}",
    "generate_non_example": "Generate a non-example of the following concept: {concept}",
    "classify": (
        "Is the following example a true instance of the concept {concept}? "
        "Answer with one word: 'yes' or 'no'.\n\n{example}"
    ),
    "num_attempts_per_concept": 10,
    "even_attempt": "non-example (true_label = 'no')",
    "odd_attempt":  "example (true_label = 'yes')",
}


def generate_incoherence_data():
    """
    Regeneration recipe for COLUMN 1 (documentation only -- needs API keys).

    Run Incoherence/main.py over CONCEPTS x MODELS, 10 attempts each, using
    COHERENCE_PROMPTS. Output CSV columns:
        Concept, Model, Example, Attempt, Inference,
        Model Label, True Label, Correct
    -> Incoherence/inferences/coherence_results.csv
    Then: incoherence = 2 * (1 - mean(Correct == 'yes')) per model
    (see recompute_incoherence_from_csv / Incoherence/incoherence_rates.py,
    verified against the paper).
    """
    raise NotImplementedError(
        "Documented recipe, not an executable run. Use Incoherence/main.py "
        "(requires API keys under AutomaticEval/private/)."
    )


# =============================================================================
# 8. CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Table 2.")
    parser.add_argument(
        "--verify", metavar="CSV", nargs="?", const=DEFAULT_INCOHERENCE_CSV,
        default=None,
        help="recompute the incoherence column from the self-consistency CSV "
             "and diff it against the published numbers. With no path, uses "
             f"{DEFAULT_INCOHERENCE_CSV}.",
    )
    args = parser.parse_args()

    print(render_latex())
    print()
    if args.verify:
        print("=" * 60)
        verify(args.verify)


if __name__ == "__main__":
    main()
