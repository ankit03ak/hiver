"""Unit tests for evaluation metrics and prediction logic."""

import pandas as pd
import pytest
from src.agent import SupportAgent
from src.evaluate import metrics, predict


@pytest.fixture
def sample_evaluation_df():
    return pd.DataFrame(
        [
            {"intent": "network_service", "predicted_intent": "network_service", "auto_handle": "yes", "predicted_handling": "auto_handle"},
            {"intent": "billing_payment", "predicted_intent": "billing_payment", "auto_handle": "no", "predicted_handling": "escalate"},
            {"intent": "account_security", "predicted_intent": "other", "auto_handle": "no", "predicted_handling": "escalate"},
            {"intent": "order_shipping", "predicted_intent": "other", "auto_handle": "yes", "predicted_handling": "escalate"},
        ]
    )


def test_metrics_calculation(sample_evaluation_df):
    df = sample_evaluation_df.copy()
    df["gold_handling"] = df["auto_handle"].map({"yes": "auto_handle", "no": "escalate"})
    res = metrics(df)
    assert res["n"] == 4
    assert res["intent_accuracy"] == 0.5
    assert res["escalation_recall"] == 1.0
    assert 0.0 <= res["intent_macro_f1"] <= 1.0


def test_predict_trivial_mode():
    agent = SupportAgent()
    res = predict("My network is down", mode="trivial", agent=agent)
    assert res["intent"] == "other"
    assert res["handling"] == "escalate"


def test_predict_simple_mode():
    agent = SupportAgent()
    res = predict("My hotspot is not working", mode="simple", agent=agent)
    assert res["intent"] == "network_service"
    assert res["handling"] == "auto_handle"
