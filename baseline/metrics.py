from typing import Dict, Iterable, List, Tuple

import numpy as np


def compute_ranking_metrics(ranks: Iterable[int]) -> Dict[str, float]:
    ranks = np.array(list(ranks), dtype=np.int64)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    mrr = np.mean(1.0 / ranks)
    return {
        "hits@1": float(hits1),
        "hits@3": float(hits3),
        "hits@10": float(hits10),
        "mrr": float(mrr),
    }


def evaluate_tail_predictions(
    queries: List[dict],
    score_fn,
    num_entities: int,
    all_triples_set: Dict[Tuple[int, int, int], None],
) -> Dict[str, float]:
    """Filtered setting evaluation for tail prediction.

    score_fn(head_ids, rel_ids, candidate_ids) -> scores (B, |candidates|)
    """
    ranks: List[int] = []

    all_entities = np.arange(num_entities, dtype=np.int64)

    for q in queries:
        h = int(q["head_id"])
        r = int(q["relation_id"])
        true_tails = set(int(t) for t in q["answers"])

        # candidate masks: filtered setting -> remove entities that are
        # valid tails for (h, r, ?) except the true target(s)
        mask = np.ones(num_entities, dtype=bool)
        for e in range(num_entities):
            if (h, r, e) in all_triples_set and e not in true_tails:
                mask[e] = False

        candidates = all_entities[mask]
        scores = score_fn(
            np.full_like(candidates, h), np.full_like(candidates, r), candidates
        )

        # For each true tail, compute its rank among filtered candidates
        for t in true_tails:
            if not mask[t]:
                # ensure true tail is in candidate set
                candidates = np.append(candidates, t)
                extra_score = score_fn(
                    np.array([h]), np.array([r]), np.array([t])
                )[0]
                scores = np.append(scores, extra_score)

            # Higher score = better rank
            # rank = 1 + #candidates with strictly higher score
            true_score = scores[candidates == t][0]

            # 特殊处理：有些模型（比如局部子图模型）会对“子图里不存在的 tail”
            # 统一返回一个非常小的分数（例如 -1e9）。如果 true tail 也是这种极小分，
            # 说明模型完全没覆盖到这个答案，此时不应该给它 rank=1，而是按最差名次计。
            if true_score <= -1e8:
                rank = int(len(candidates))
            else:
                rank = int(1 + np.sum(scores > true_score))

            ranks.append(rank)

    return compute_ranking_metrics(ranks)

