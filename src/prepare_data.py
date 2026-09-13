"""Create a compact, reproducible T-Mobile conversation subset from TWCS."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


BRAND = "TMobileHelp"
SEED = "hiver-tmobile-v1"


def stable_bucket(tweet_id: int) -> int:
    digest = hashlib.sha256(f"{SEED}:{tweet_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to twcs.csv or twcs.csv.zip")
    parser.add_argument("--output", default="data/processed/tmobile_pairs.csv")
    parser.add_argument("--max-pairs", type=int, default=12_000)
    args = parser.parse_args()

    reply_by_parent: dict[int, str] = {}
    columns = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
    for chunk in pd.read_csv(args.input, usecols=columns, chunksize=200_000):
        brand_replies = chunk.loc[
            chunk["author_id"].eq(BRAND) & chunk["in_response_to_tweet_id"].notna(),
            ["in_response_to_tweet_id", "text"],
        ]
        for parent_id, reply in brand_replies.itertuples(index=False):
            reply_by_parent[int(parent_id)] = str(reply)

    customers: dict[int, str] = {}
    for chunk in pd.read_csv(args.input, usecols=["tweet_id", "inbound", "text"], chunksize=200_000):
        matching = chunk.loc[
            chunk["tweet_id"].isin(reply_by_parent) & chunk["inbound"].eq(True),
            ["tweet_id", "text"],
        ]
        for tweet_id, text in matching.itertuples(index=False):
            customers[int(tweet_id)] = str(text)

    pairs = pd.DataFrame(
        {
            "customer_tweet_id": list(customers),
            "customer_text": list(customers.values()),
        }
    )
    pairs["historical_reply"] = pairs["customer_tweet_id"].map(reply_by_parent)
    pairs = pairs.dropna().drop_duplicates("customer_tweet_id")
    pairs["bucket"] = pairs["customer_tweet_id"].map(stable_bucket)
    pairs["split"] = pairs["bucket"].map(
        lambda bucket: "golden_pool" if bucket < 2 else "test" if bucket < 7 else "train"
    )
    pairs = pairs.sort_values("customer_tweet_id")

    # Cap each split proportionally after deterministic split, retaining repeatability.
    selected = []
    for split, limit in {"train": int(args.max_pairs * 0.93), "test": int(args.max_pairs * 0.05), "golden_pool": int(args.max_pairs * 0.02)}.items():
        selected.append(pairs.loc[pairs["split"].eq(split)].head(limit))
    output = pd.concat(selected, ignore_index=True).drop(columns=["bucket"])

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Wrote {len(output):,} direct customer-to-{BRAND} pairs to {path}")
    print(output["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
