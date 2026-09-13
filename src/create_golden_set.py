"""Create blind human-annotation and hidden-reference files from held-out cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/processed/tmobile_pairs.csv")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--annotation-output", default="data/golden/annotation_template.csv")
    parser.add_argument("--reference-output", default="data/golden/reference_replies.csv")
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    pool = pairs.loc[pairs["split"].eq("golden_pool")].sample(args.count, random_state=20260909)
    pool = pool.sort_values("customer_tweet_id").reset_index(drop=True)
    pool.insert(0, "example_id", [f"golden-{index:03d}" for index in range(1, len(pool) + 1)])

    annotation = pool[["example_id", "customer_tweet_id", "customer_text"]].copy()
    annotation["intent"] = ""
    annotation["auto_handle"] = ""
    annotation["escalation_reason"] = ""
    annotation["notes"] = ""
    annotation["annotation_status"] = "unreviewed"

    reference = pool[["example_id", "customer_tweet_id", "historical_reply"]].copy()
    for target in (args.annotation_output, args.reference_output):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(args.annotation_output, index=False)
    reference.to_csv(args.reference_output, index=False)
    print(f"Created {len(annotation)} blind annotation rows at {args.annotation_output}")
    print(f"Created hidden historical references at {args.reference_output}")


if __name__ == "__main__":
    main()
