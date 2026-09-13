"""Unit and integration tests for SupportAgent logic, risk flags, and fallback intent."""

from unittest.mock import MagicMock, patch
import pytest
from src.agent import AgentResult, SupportAgent, fallback_intent, risk_flags


def test_risk_flags_detection():
    # Fraud & account security
    assert "possible account fraud or identity issue" in risk_flags("Someone opened an unauthorized account in my name!")
    assert "account-security or PIN concern" in risk_flags("I forgot my passcode and password and am locked out.")

    # Payment dispute
    assert "payment dispute or refund request" in risk_flags("I was charged twice on my credit card and need a refund.")

    # Unresolved support & outage
    assert "repeated unresolved support failure" in risk_flags("I am still waiting for a response for the second time.")
    assert "service outage or business-critical connectivity" in risk_flags("Major outage in my area, I cannot work at all.")

    # Safe message
    assert len(risk_flags("How do I check my monthly data balance on my account?")) == 0


def test_fallback_intent_classification():
    assert fallback_intent("Someone opened a fake account with my pin") == "account_security"
    assert fallback_intent("My bill and payment charge are wrong") == "billing_payment"
    assert fallback_intent("No signal or 4G LTE network coverage") == "network_service"
    assert fallback_intent("Want to upgrade my handset to a new phone") == "device_upgrade_unlock"
    assert fallback_intent("Is Netflix included with my plan promo?") == "plan_features_promotions"
    assert fallback_intent("Where is my order shipping tracking code?") == "order_shipping"
    assert fallback_intent("Hello good morning T-Mobile") == "other"


def test_support_agent_respond_offline_mode():
    agent = SupportAgent()
    agent.api_key = None  # Force offline fallback mode

    # Safe message
    result = agent.respond("How can I check my data usage on the plan?")
    assert isinstance(result, AgentResult)
    assert result.mode == "offline fallback"
    assert result.handling == "auto_handle"
    assert result.intent in ["plan_features_promotions", "network_service", "other"]

    # High-risk message
    risk_result = agent.respond("I suspect fraud on my account and need a refund.")
    assert risk_result.handling == "escalate"
    assert "fraud" in risk_result.escalation_reason.lower() or "payment" in risk_result.escalation_reason.lower()
    assert "DM" in risk_result.reply


@patch("google.genai.Client")
def test_support_agent_gemini_mode_parsing(mock_client_cls):
    agent = SupportAgent()
    agent.api_key = "test_mock_key"
    agent.model = "gemini-2.5-flash"

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = '```json\n{"intent": "network_service", "handling": "auto_handle", "escalation_reason": "", "reply": "We can help you troubleshoot your network settings. Please send us a DM.", "evidence_case_numbers": [1]}\n```'
    mock_client.models.generate_content.return_value = mock_response

    result = agent.respond("My hotspot internet connection is very slow today.")
    assert result.intent == "network_service"
    assert result.handling == "auto_handle"
    assert "DM" in result.reply
    assert result.mode == "Gemini (gemini-2.5-flash)"
