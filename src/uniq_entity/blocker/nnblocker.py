from __future__ import annotations

import pandas as pd
from retriv import DenseRetriever, HybridRetriever, SparseRetriever


class NNBlocker:
    def __init__(
        self,
        retriever_class: type[SparseRetriever | DenseRetriever | HybridRetriever],
        **retriever_kwargs,
    ) -> None:
        self.retriever = retriever_class(**retriever_kwargs)

    def index(
        self,
        data: pd.DataFrame,
        record2text: callable,
        **index_kwargs,
    ) -> None:
        self.record2text = record2text
        collection = [
            {"id": str(k), "text": self.record2text(v)}
            for k, v in data.to_dict(orient="index").items()
        ]
        self.retriever.index(collection, **index_kwargs)

    def search(
        self,
        query: str | dict,
        topk: int = 10,
    ) -> dict:
        if isinstance(query, dict):
            query = self.callback(query)

        return self.retriever.search(query, cutoff=topk, return_docs=False)

    def batch_search(
        self,
        queries: list[dict],
        topk: int = 10,
    ) -> dict[dict]:
        results = self.retriever.bsearch(queries, cutoff=topk)

        return results

    def join(
        self,
        data: pd.DataFrame,
        topk: int = 10,
    ) -> dict[dict]:
        collection = [
            {"id": str(k), "text": self.record2text(v)}
            for k, v in data.to_dict(orient="index").items()
        ]

        return self.batch_search(collection, topk=topk)
