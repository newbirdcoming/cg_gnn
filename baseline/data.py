import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class KGData:
    num_entities: int
    num_relations: int
    train_triples: np.ndarray  # (N_train, 3)
    valid_triples: np.ndarray  # (N_valid, 3)
    test_triples: np.ndarray  # (N_test, 3)
    valid_queries: List[dict]
    test_queries: List[dict]
    all_triples_set: Dict[Tuple[int, int, int], None]


def load_processed_data(data_dir: Path) -> KGData:
    processed = data_dir / "processed"

    with (processed / "entity2id.json").open("r", encoding="utf-8") as f:
        entity2id: Dict[str, int] = json.load(f)
    with (processed / "relation2id.json").open("r", encoding="utf-8") as f:
        relation2id: Dict[str, int] = json.load(f)

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    def load_txt_split(name: str) -> np.ndarray:
        path = processed / f"{name}.txt"
        triples: List[Tuple[int, int, int]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h, r, t = line.split("\t")
                triples.append((int(h), int(r), int(t)))
        return np.array(triples, dtype=np.int64)

    train_triples = load_txt_split("train")
    valid_triples = load_txt_split("valid")
    test_triples = load_txt_split("test")

    with (processed / "valid_queries.json").open("r", encoding="utf-8") as f:
        valid_queries = json.load(f)
    with (processed / "test_queries.json").open("r", encoding="utf-8") as f:
        test_queries = json.load(f)

    # all triples for filtered setting
    all_triples = np.concatenate([train_triples, valid_triples, test_triples], axis=0)
    all_triples_set: Dict[Tuple[int, int, int], None] = {
        (int(h), int(r), int(t)): None for h, r, t in all_triples
    }

    return KGData(
        num_entities=num_entities,
        num_relations=num_relations,
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        valid_queries=valid_queries,
        test_queries=test_queries,
        all_triples_set=all_triples_set,
    )

