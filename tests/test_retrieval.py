"""Unit tests for TF-IDF historical case retriever."""

import pandas as pd
import pytest
from src.retrieval import HistoricalRetriever, RetrievedCase


@pytest.fixture
def sample_cases():
    return pd.DataFrame(
        [
            {
                "customer_tweet_id": 101,
                "customer_text": "My hotspot is down and not connecting.",
                "historical_reply": "We can help reset your hotspot settings via DM.",
                "split": "train",
            },
            {
                "customer_tweet_id": 102,
                "customer_text": "Need help with my monthly billing charge.",
                "historical_reply": "Send us a DM with your account details to check the bill.",
                "split": "train",
            },
            {
                "customer_tweet_id": 103,
                "customer_text": "Hotspot internet connection speed is very slow.",
                "historical_reply": "Please check your network signal or send us a DM.",
                "split": "train",
            },
        ]
    )


def test_retriever_initialization(sample_cases):
    retriever = HistoricalRetriever(sample_cases)
    assert len(retriever.cases) == 3
    assert retriever.matrix.shape[0] == 3


def test_retriever_search_returns_top_k(sample_cases):
    retriever = HistoricalRetriever(sample_cases)
    results = retriever.search("hotspot connection issues", k=2)
    assert len(results) == 2
    assert isinstance(results[0], RetrievedCase)
    # The top result should be related to hotspot (ID 101 or 103)
    assert results[0].customer_tweet_id in [101, 103]
    assert results[0].score >= results[1].score


def test_retriever_empty_or_unknown_query(sample_cases):
    retriever = HistoricalRetriever(sample_cases)
    results = retriever.search("xyz unknown vocabulary term", k=1)
    assert len(results) == 1
    assert results[0].score == 0.0
