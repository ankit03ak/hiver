"""Evaluate intent and routing against the manually completed golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.agent import SupportAgent, fallback_intent, risk_flags


def predict(message: str, mode: str, agent: SupportAgent) -> dict[str, str]:
    if mode == "trivial":
        return {"intent": "other", "handling": "escalate", "reply": "Please send us a DM so we can help."}
    if mode == "simple":
        flags = risk_flags(message)
        evidence = agent.retriever.search(message, k=1)
        reply = evidence[0].historical_reply if evidence else "Please send us a DM so we can help."
        return {
            "intent": fallback_intent(message),
            "handling": "escalate" if flags else "auto_handle",
            "reply": reply,
        }
    result = agent.respond(message)
    return {"intent": result.intent, "handling": result.handling, "reply": result.reply}


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    intent_true = frame["intent"].astype(str)
    intent_pred = frame["predicted_intent"].astype(str)
    routing_true = frame["gold_handling"].astype(str)
    routing_pred = frame["predicted_handling"].astype(str)
    precision, recall, f1, _ = precision_recall_fscore_support(
        routing_true, routing_pred, pos_label="escalate", average="binary", zero_division=0
    )
    res = {
        "n": len(frame),
        "intent_accuracy": round(float(accuracy_score(intent_true, intent_pred)), 4),
        "intent_macro_f1": round(float(f1_score(intent_true, intent_pred, average="macro", zero_division=0)), 4),
        "escalation_precision": round(float(precision), 4),
        "escalation_recall": round(float(recall), 4),
        "escalation_f1": round(float(f1), 4),
    }
    if "reply_quality" in frame.columns:
        res["reply_quality_overall"] = round(float(frame["reply_quality"].mean()), 4)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/golden/annotation_template.csv")
    parser.add_argument("--mode", choices=["trivial", "simple", "gemini"], default="gemini")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    labels = pd.read_csv(args.annotations, keep_default_na=False)
    labels = labels.loc[labels["annotation_status"].eq("complete")].copy()
    if labels.empty:
        raise SystemExit("No completed annotations. Use label_golden.py before evaluating.")
    if args.limit:
        labels = labels.head(args.limit)
    invalid = labels.loc[~labels["auto_handle"].isin(["yes", "no"]) ]
    if not invalid.empty:
        raise SystemExit("auto_handle must be yes or no for every completed row.")

    agent = SupportAgent()
    predictions = [predict(text, args.mode, agent) for text in labels["customer_text"]]
    result = labels.copy()
    result["gold_handling"] = result["auto_handle"].map({"yes": "auto_handle", "no": "escalate"})
    result["predicted_intent"] = [item["intent"] for item in predictions]
    result["predicted_handling"] = [item["handling"] for item in predictions]
    result["draft_reply"] = [item["reply"] for item in predictions]

    from src.judge import ReplyJudge
    judge = ReplyJudge()
    scores = [judge.evaluate_reply(cust, draft) for cust, draft in zip(result["customer_text"], result["draft_reply"])]
    result["reply_quality"] = [s.overall for s in scores]
    result["reply_safety"] = [s.safety for s in scores]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    result.to_csv(output_dir / f"predictions_{args.mode}.csv", index=False)
    summary = metrics(result)
    (output_dir / f"metrics_{args.mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
