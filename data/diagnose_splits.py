"""
诊断脚本：检查 train / valid / test 上的「tail 可达性」。

指标：
  1. tail 是否在背景图（background.txt）里出现过（节点级别）
  2. tail 是否能通过风险因果链从 complaint 节点 BFS 可达（链路级别）

使用纯 Python BFS 代替 extract_local_subgraph，大幅提速。
"""

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────── 工具函数 ───────────────────────────

def load_triples(path: Path) -> np.ndarray:
    rows: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rows.append((int(h), int(r), int(t)))
    if not rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def build_adj(triples: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    """head_id -> [(rel_id, tail_id)]"""
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        if h not in adj:
            adj[h] = []
        adj[h].append((r, t))
    return adj


def reachable_nodes_bfs(
    start: int,
    adj: Dict[int, List[Tuple[int, int]]],
    allowed_rels: Set[int],
    max_hops: int = 4,
) -> Set[int]:
    """
    从 start 出发，只沿 allowed_rels 里的边做 BFS，
    返回可达节点集合（含起点）。
    max_hops 防止超大图耗时过长。
    """
    visited: Set[int] = {start}
    queue: deque = deque([(start, 0)])
    while queue:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for r, t in adj.get(node, []):
            if r in allowed_rels and t not in visited:
                visited.add(t)
                queue.append((t, hops + 1))
    return visited


# ─────────────────────────── 主逻辑 ───────────────────────────

def main() -> None:
    relation2id: Dict[str, int] = json.loads(
        (PROCESSED / "relation2id.json").read_text(encoding="utf-8")
    )

    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    rel_inc_entity = relation2id.get("包含实体")
    rel_inc_hidden = relation2id.get("包含隐患")
    rel_inc_event = relation2id.get("包含事件")
    rel_leads_to = relation2id.get("导致")
    rel_susceptible = relation2id.get("易感于")
    rel_trigger = relation2id.get("触发风险")

    # 风险因果链所允许的关系集合（与 subgraph.py 保持一致）
    causal_rels: Set[int] = set(filter(None, [
        rel_inc_entity, rel_inc_hidden, rel_inc_event,
        rel_leads_to, rel_susceptible, rel_trigger,
    ]))

    # ── 邻接表：优先使用 background.txt ──
    background_path = PROCESSED / "background.txt"
    if background_path.exists():
        print(f"[diagnose] Using background.txt ({background_path})")
        bg_triples = load_triples(background_path)
    else:
        print("[diagnose] background.txt not found, using train.txt only")
        bg_triples = load_triples(PROCESSED / "train.txt")

    adj = build_adj(bg_triples)
    bg_entities: Set[int] = set(int(x) for x in bg_triples[:, 0]) | set(
        int(x) for x in bg_triples[:, 2]
    )
    print(f"[diagnose] background graph: {len(bg_triples)} triples, "
          f"{len(bg_entities)} unique entities\n")

    # ── train 目标三元组的子图覆盖率 ──
    train_triples = load_triples(PROCESSED / "train.txt")
    mask = np.zeros(len(train_triples), dtype=bool)
    if rel_risk is not None:
        mask |= train_triples[:, 1] == rel_risk
    if rel_outcome is not None:
        mask |= train_triples[:, 1] == rel_outcome

    target_train = train_triples[mask]
    total_train = len(target_train)
    in_subgraph_train = 0

    for h, r, t in target_train:
        reachable = reachable_nodes_bfs(int(h), adj, causal_rels)
        if int(t) in reachable:
            in_subgraph_train += 1

    ratio_tr = in_subgraph_train / total_train if total_train else 0
    print("=" * 50)
    print(f"TRAIN: target triples = {total_train}")
    print(f"TRAIN: tail reachable via causal chain = {in_subgraph_train} "
          f"({ratio_tr:.2%})")

    # ── valid / test 统计 ──
    for split in ("valid", "test"):
        queries: List[dict] = json.loads(
            (PROCESSED / f"{split}_queries.json").read_text(encoding="utf-8")
        )

        total_q = 0
        tail_in_bg = 0
        tail_reachable = 0

        for q in queries:
            h = int(q["head_id"])
            r = int(q["relation_id"])
            if r not in (rel_risk, rel_outcome):
                continue

            answers = [int(t) for t in q["answers"]]
            reachable = reachable_nodes_bfs(int(h), adj, causal_rels)

            for t in answers:
                total_q += 1
                if t in bg_entities:
                    tail_in_bg += 1
                if t in reachable:
                    tail_reachable += 1

        ratio_bg = tail_in_bg / total_q if total_q else 0
        ratio_sub = tail_reachable / total_q if total_q else 0

        print()
        print("=" * 50)
        print(f"{split.upper()}: total query answers = {total_q}")
        print(f"{split.upper()}: tail seen in background graph = {tail_in_bg} "
              f"({ratio_bg:.2%})")
        print(f"{split.upper()}: tail reachable via causal chain = {tail_reachable} "
              f"({ratio_sub:.2%})")

    print()
    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
