from __future__ import annotations

import pandas as pd
from retriv import DenseRetriever, HybridRetriever, SparseRetriever
from sklearn.metrics import auc


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

    def evaluate(
        self,
        blocks: dict[dict],
        matches: set[tuple],
        *,
        return_list: bool = False,
    ) -> dict:
        for k, v in blocks.items():
            blocks[k] = sorted(v, key=v.get, reverse=True)

        topk = max(len(dct) for dct in blocks.values())
        candidates = []
        flags = set()  # Comparison Propagation
        for k in range(topk):
            cands = set()
            for i in blocks:
                if k >= len(blocks[i]):
                    continue

                j = blocks[i][k]
                pair = i, j
                if i != j and pair not in flags:
                    cands.add(pair)
                    flags.add(pair)

            candidates.append(cands)

        cands = set()
        precisions, recalls = [1], [0]
        for k in range(topk):
            cands |= candidates[k]
            tp = len(cands & matches)
            ppv = tp / len(cands)
            tpr = tp / len(matches)
            precisions.append(ppv)
            recalls.append(tpr)

        mAP = auc(recalls, precisions)
        if not return_list:
            precision = precisions[-1]
            recall = recalls[-1]
            f1 = 2 * (precision * recall) / (precision + recall)
            return {"mAP": mAP, "Recall": recall, "Precision": precision, "F1": f1}
        else:
            return {"mAP": mAP, "Recalls": recalls, "Precisions": precisions}
