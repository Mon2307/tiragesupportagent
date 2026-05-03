"""
main.py — Batch runner for the support-triage agent.

Reads support_tickets/support_tickets.csv, runs every ticket through
run_agent(), and writes results to support_tickets/output.csv.

Usage:
    cd code/
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running from either the repo root or the code/ directory.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from retriever import load_articles
from agent import run_agent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_CSV = _REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = _REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load the corpus once — reused for every ticket.
    print("Loading corpus …", flush=True)
    corpus = load_articles()
    print(f"Corpus ready: {len(corpus)} articles.\n", flush=True)

    # Read tickets — normalise column names to lowercase so the CSV header
    # capitalisation ("Issue" vs "issue") never causes silent empty reads.
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip().str.lower()
    total = len(df)
    print(f"Found {total} tickets in {INPUT_CSV.name}.", flush=True)
    print(f"Columns detected: {list(df.columns)}\n", flush=True)

    results: list[dict] = []

    for idx, row in df.iterrows():
        row_num = int(idx) + 1  # type: ignore[arg-type]
        # print(f"Processing row {row_num}/{total} …", flush=True)

        issue   = str(row.get("issue",   "") or "").strip()
        subject = str(row.get("subject", "") or "").strip()
        company = str(row.get("company", "") or "").strip()

        # print(f"  company={company!r}  subject={subject[:60]!r}", flush=True)

        try:
            agent_out = run_agent(
                issue=issue,
                subject=subject,
                company=company,
                corpus=corpus,
            )
        except Exception as exc:  # noqa: BLE001
            # print(f"  [ERROR] row {row_num} failed: {type(exc).__name__}: {exc}", flush=True)
            agent_out = {
                "status": "escalated",
                "product_area": "general_support",
                "response": "An unexpected error occurred while processing this ticket.",
                "justification": f"Exception during agent run: {type(exc).__name__}: {exc}",
                "request_type": "product_issue",
            }

        # print(
        #     f"  → status={agent_out['status']}  "
        #     f"type={agent_out['request_type']}  "
        #     f"area={agent_out['product_area']}",
        #     flush=True,
        # )

        results.append(
            {
                "issue": issue,
                "subject": subject,
                "company": company,
                **agent_out,
            }
        )

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
