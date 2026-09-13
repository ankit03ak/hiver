"""Lexical retrieval of historically resolved T-Mobile support cases."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedCase:
    customer_tweet_id: int
    customer_text: str
    historical_reply: str
    score: float


class HistoricalRetriever:
    def __init__(self, cases: pd.DataFrame) -> None:
        self.cases = cases.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.cases["customer_text"].fillna(""))

    def search(self, message: str, k: int = 3) -> list[RetrievedCase]:
        query = self.vectorizer.transform([message])
        scores = cosine_similarity(query, self.matrix).ravel()
        best_indices = scores.argsort()[-k:][::-1]
        return [
            RetrievedCase(
                customer_tweet_id=int(self.cases.iloc[index]["customer_tweet_id"]),
                customer_text=str(self.cases.iloc[index]["customer_text"]),
                historical_reply=str(self.cases.iloc[index]["historical_reply"]),
                score=float(scores[index]),
            )
            for index in best_indices
        ]
