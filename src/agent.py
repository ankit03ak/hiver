"""Grounded T-Mobile support agent with a Gemini and an offline fallback mode."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.retrieval import HistoricalRetriever, RetrievedCase


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = json.loads((ROOT / "config" / "intent_taxonomy.json").read_text(encoding="utf-8"))


@dataclass
class AgentResult:
    intent: str
    handling: str
    escalation_reason: str
    reply: str
    evidence: list[RetrievedCase]
    mode: str


def risk_flags(message: str) -> list[str]:
    lowered = message.lower()
    patterns = {
        "possible account fraud or identity issue": r"fraud|unauthori[sz]ed|someone opened|identity theft|stolen identity",
        "account-security or PIN concern": r"password|pin|passcode|locked out|account access",
        "payment dispute or refund request": r"chargeback|dispute|refund|charged twice|incorrect charge",
        "repeated unresolved support failure": r"still waiting|no one.*respond|again|second time|third time|\bweeks?\b",
        "service outage or business-critical connectivity": r"outage|emergency|can't work|cannot work|no service.*hours",
    }
    return [reason for reason, pattern in patterns.items() if re.search(pattern, lowered)]


def fallback_intent(message: str) -> str:
    lowered = message.lower()
    if re.search(r"fraud|unauthori[sz]ed|someone opened|identity theft|stolen identity|password|\bpin\b", lowered):
        return "account_security"
    keywords = {
        "account_security": ["fraud", "unauthorized", "password", "pin", "account access", "identity"],
        "billing_payment": ["bill", "charge", "fee", "payment", "refund", "card"],
        "network_service": ["signal", "network", "lte", "data", "hotspot", "coverage", "service"],
        "device_upgrade_unlock": ["iphone", "phone", "device", "upgrade", "trade", "unlock", "activation"],
        "plan_features_promotions": ["plan", "promotion", "promo", "netflix", "discount", "eligible"],
        "order_shipping": ["order", "shipping", "ship", "delivery", "pre-order", "missing"],
        "support_follow_up": ["waiting", "respond", "dm", "representative", "support"],
    }
    return max(keywords, key=lambda intent: sum(word in lowered for word in keywords[intent])) if any(
        word in lowered for words in keywords.values() for word in words
    ) else "other"


class SupportAgent:
    def __init__(self, pairs_path: str | Path = ROOT / "data" / "processed" / "tmobile_pairs.csv") -> None:
        cases = pd.read_csv(pairs_path)
        self.retriever = HistoricalRetriever(cases.loc[cases["split"].eq("train")])
        load_dotenv(ROOT / ".env")
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    def respond(self, message: str) -> AgentResult:
        evidence = self.retriever.search(message)
        flags = risk_flags(message)
        if not self.api_key:
            return self._fallback(message, evidence, flags)
        return self._gemini(message, evidence, flags)

    def _fallback(self, message: str, evidence: list[RetrievedCase], flags: list[str]) -> AgentResult:
        intent = fallback_intent(message)
        handling = "escalate" if flags else "auto_handle"
        if handling == "escalate":
            reason = "; ".join(flags)
            reply = "I’m sorry you’re dealing with this. For your privacy, please send us a DM so a specialist can securely review your account."
        else:
            reason = "No high-risk escalation trigger detected."
            reply = evidence[0].historical_reply if evidence else "Thanks for reaching out. Please send us a DM so we can take a closer look."
        return AgentResult(intent, handling, reason, reply, evidence, "offline fallback")

    def _gemini(self, message: str, evidence: list[RetrievedCase], flags: list[str]) -> AgentResult:
        from google import genai

        evidence_text = "\n\n".join(
            f"CASE {index + 1} (similarity={case.score:.2f})\nCustomer: {case.customer_text}\nHistorical reply: {case.historical_reply}"
            for index, case in enumerate(evidence)
        )
        force_escalation = bool(flags)
        prompt = f"""You are a cautious T-Mobile social-support assistant. Return only valid JSON with keys intent, handling, escalation_reason, reply, evidence_case_numbers.

Allowed intents: {json.dumps(TAXONOMY)}
Customer message: {message}
Risk flags from deterministic policy: {json.dumps(flags)}
Historical cases (use only these as support-style evidence):
{evidence_text}

Rules:
- handling must be auto_handle or escalate.
- If risk flags are non-empty, handling must be escalate and state the most relevant flag.
- Never ask for personal, payment, PIN, or account details publicly; direct the customer to DM for account-specific help.
- Do not invent policy, pricing, eligibility, time estimates, or resolution status.
- Keep reply under 280 characters, empathetic, and practical.
- Cite only case numbers that support the reply."""
        client = genai.Client(api_key=self.api_key)
        try:
            response = client.models.generate_content(model=self.model, contents=prompt)
        except Exception as error:
            fallback = self._fallback(message, evidence, flags)
            fallback.mode = f"offline fallback (Gemini unavailable: {type(error).__name__})"
            return fallback
        try:
            raw_text = response.text.strip() if response.text else ""
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            intent = parsed.get("intent") if parsed.get("intent") in TAXONOMY else "other"
            handling = parsed.get("handling")
            if force_escalation:
                handling = "escalate"
            if handling not in {"auto_handle", "escalate"}:
                handling = "escalate"
            reply = str(parsed.get("reply", "Thanks for reaching out. Please send us a DM so we can take a closer look."))[:280]
            return AgentResult(
                intent=intent,
                handling=handling,
                escalation_reason=str(parsed.get("escalation_reason", "")),
                reply=reply,
                evidence=evidence,
                mode=f"Gemini ({self.model})",
            )
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return self._fallback(message, evidence, flags)


def result_as_dict(result: AgentResult) -> dict:
    payload = asdict(result)
    payload["evidence"] = [asdict(case) for case in result.evidence]
    return payload
