import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"


def load_mappings():
    entity2id = json.loads((PROCESSED / "entity2id.json").read_text(encoding="utf-8"))
    id2entity = {v: k for k, v in entity2id.items()}
    relation2id = json.loads(
        (PROCESSED / "relation2id.json").read_text(encoding="utf-8")
    )
    id2rel = {v: k for k, v in relation2id.items()}
    return id2entity, id2rel


def load_triples(name: str = "train"):
    triples = []
    with (PROCESSED / f"{name}.txt").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            triples.append((int(h), int(r), int(t)))
    return np.array(triples, dtype=int)


def main():
    id2entity, id2rel = load_mappings()
    triples = load_triples("train")

    neighbors = defaultdict(list)
    for h, r, t in triples:
        neighbors[h].append((r, t))

    node_ids = {h for h, _, _ in triples} | {t for _, _, t in triples}
    print("train graph num_nodes:", len(node_ids))
    print("train graph num_edges:", len(triples))

    print("\nSample: a few complaint nodes and their outgoing edges (IDs only, no Chinese labels):")
    count = 0
    for ent_id, label in id2entity.items():
        if not label.startswith("complaint:"):
            continue
        print(f"\nNode {ent_id} outgoing edges (label={label}):")
        for r, t in neighbors.get(ent_id, [])[:20]:
            # 这里只打印 id，避免中文关系名在控制台乱码
            print(f"  - rel_id={r} -> node_id={t}")
        count += 1
        if count >= 5:
            break


if __name__ == "__main__":
    main()

