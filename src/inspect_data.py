"""Inspect telecom candidates in the Customer Support on Twitter dataset.

Reads the Kaggle ZIP directly, so the large raw CSV never needs extracting.
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd


TELECOM_ACCOUNTS = {"TMobileHelp", "sprintcare", "VerizonSupport", "O2", "ATT"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to twcs.csv or twcs.csv.zip")
    args = parser.parse_args()

    outbound_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(args.input, usecols=["author_id", "inbound"], chunksize=200_000):
        company_rows = chunk.loc[chunk["inbound"].eq(False), "author_id"].dropna()
        outbound_counts.update(company_rows.astype(str))

    print("Telecom candidate volumes (brand-authored tweets):")
    for account in sorted(TELECOM_ACCOUNTS, key=outbound_counts.get, reverse=True):
        print(f"  {account:16} {outbound_counts[account]:,}")


if __name__ == "__main__":
    main()
