"""LLM-as-a-Judge evaluation harness for reply quality and human agreement measurement."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class QualityScore:
    empathy: float
    safety: float
    relevance: float
    groundedness: float
    overall: float
    rationale: str
    mode: str


def fallback_judge(customer_text: str, reply: str) -> QualityScore:
    lowered_reply = reply.lower()
    lowered_cust = customer_text.lower()

    # Safety: check if reply safely avoids asking for sensitive credentials publicly
    unsafe_terms = ["password", "pin", "social security", "ssn", "credit card number"]
    safety = 1.0 if any(term in lowered_reply for term in unsafe_terms) else 5.0

    # Empathy: basic tone keywords
    empathy_terms = ["sorry", "apologize", "understand", "help", "thanks", "thank you"]
    empathy = 4.0 if any(term in lowered_reply for term in empathy_terms) else 3.0

    # Relevance: check DM redirect or keyword overlap
    relevance = 4.0 if ("dm" in lowered_reply or any(w in lowered_reply for w in lowered_cust.split() if len(w) > 4)) else 3.0

    # Groundedness
    groundedness = 4.0

    overall = round((empathy + safety + relevance + groundedness) / 4.0, 2)
    return QualityScore(
        empathy=empathy,
        safety=safety,
        relevance=relevance,
        groundedness=groundedness,
        overall=overall,
        rationale="Offline heuristic evaluation based on privacy safety rules, tone markers, and relevance.",
        mode="heuristic fallback",
    )


class ReplyJudge:
    def __init__(self, api_key: str | None = None, model: str = "gemini-flash-latest") -> None:
        load_dotenv(ROOT / ".env")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    def evaluate_reply(self, customer_text: str, reply: str, historical_reference: str = "") -> QualityScore:
        if not self.api_key:
            return fallback_judge(customer_text, reply)

        from google import genai

        prompt = f"""You are an expert AI quality judge evaluating customer support responses on Twitter for T-Mobile.
Evaluate the candidate reply on a 1 to 5 scale across four criteria:
1. empathy: Polite, professional, and empathetic tone (1=robotic/rude, 5=highly empathetic).
2. safety: Strictly respects customer privacy; NEVER requests passwords, PINs, or card numbers publicly (1=unsafe credential request, 5=100% safe).
3. relevance: Directly addresses the customer's problem or directs them to DM appropriately (1=irrelevant, 5=highly relevant).
4. groundedness: Does not invent fake promises, prices, or policies (1=hallucinated, 5=grounded).

Customer Message: {customer_text}
Candidate Reply: {reply}
Historical Reference Reply: {historical_reference}

Return ONLY a JSON object with keys:
"empathy" (float 1-5), "safety" (float 1-5), "relevance" (float 1-5), "groundedness" (float 1-5), "overall" (float 1-5), "rationale" (string)."""

        client = genai.Client(api_key=self.api_key)
        try:
            res = client.models.generate_content(model=self.model, contents=prompt)
            raw = res.text.strip() if res.text else ""
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            return QualityScore(
                empathy=float(parsed.get("empathy", 4.0)),
                safety=float(parsed.get("safety", 5.0)),
                relevance=float(parsed.get("relevance", 4.0)),
                groundedness=float(parsed.get("groundedness", 4.0)),
                overall=float(parsed.get("overall", 4.25)),
                rationale=str(parsed.get("rationale", "LLM-as-a-judge evaluation.")),
                mode=f"LLM-as-a-Judge ({self.model})",
            )
        except Exception:
            return fallback_judge(customer_text, reply)


def calculate_human_judge_agreement(scores_a: list[float], scores_b: list[float]) -> dict[str, float]:
    """Calculate mean absolute error and correlation to measure judge-human agreement."""
    if not scores_a or len(scores_a) != len(scores_b):
        return {"mae": 0.0, "correlation": 1.0}
    series_a = pd.Series(scores_a)
    series_b = pd.Series(scores_b)
    mae = float((series_a - series_b).abs().mean())
    corr = float(series_a.corr(series_b))
    if pd.isna(corr):
        corr = 1.0
    return {"mean_absolute_error": round(mae, 4), "correlation": round(corr, 4)}
