import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.encoder import LocalRGCNEncoder
from subgraph_model.decoder import SubgraphDistMultDecoder
from subgraph_model.subgraph import build_adjacency, extract_local_subgraph
from subgraph_model.subgraph_dynamic import extract_dynamic_subgraph, dynamic_path_support_mapping


class MinimalSubgraphModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = LocalRGCNEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            num_layers=num_layers,
        )
        self.decoder = SubgraphDistMultDecoder(num_relations=num_relations, dim=dim)

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        global2local: Dict[int, int],
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        z_local = self.encoder(node_ids, edge_index, edge_type)
        return self.decoder(z_local, global2local, heads, rels, tails)


STRICT_EVAL_DYNAMIC = False


def _load_adjacency(processed: Path) -> Dict[int, List[Tuple[int, int]]]:
    """加载背景图邻接表（优先用 background.txt，否则退回 train.txt）。"""
    bg_path = processed / "background.txt"
    src_path = bg_path if bg_path.exists() else processed / "train.txt"
    label = "background.txt" if bg_path.exists() else "train.txt (fallback)"
    print(f"[MinimalDynamic] Adjacency source: {label}", flush=True)

    rows: List[Tuple[int, int, int]] = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rows.append((int(h), int(r), int(t)))
    return build_adjacency(np.array(rows, dtype=np.int64))


def _load_entity_type_sets(processed: Path) -> Tuple[List[int], List[int]]:
    """返回 risk / outcome 的全局 entity id 列表。"""
    entity2id = json.loads((processed / "entity2id.json").read_text(encoding="utf-8"))
    risk_entities = [eid for name, eid in entity2id.items() if name.startswith("risk:")]
    outcome_entities = [eid for name, eid in entity2id.items() if name.startswith("outcome:")]
    return risk_entities, outcome_entities


def _invert_relation2id(relation2id: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in relation2id.items()}


def _print_debug_dynamic_subgraph(
    kg,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
    num_entities: int,
    device: str,
) -> None:
    """按验证要求打印：随机风险/后果样本的动态子图，并与旧版固定子图对比。"""
    risk_rel_id = relation2id.get("包含风险")
    outcome_rel_id = relation2id.get("包含后果")
    id2rel = _invert_relation2id(relation2id)

    def _pretty_edges(edge_index: torch.Tensor, edge_type: torch.Tensor, global2local: Dict[int, int]) -> List[str]:
        # edge_index: (2, E) uses local indices; we output global ids for readability.
        # Reverse local->global:
        local2global = {v: k for k, v in global2local.items()}
        lines: List[str] = []
        for e in range(edge_index.size(1)):
            src_l = int(edge_index[0, e].item())
            dst_l = int(edge_index[1, e].item())
            src_g = int(local2global[src_l])
            dst_g = int(local2global[dst_l])
            rel_g = int(edge_type[e].item())
            lines.append(f"{src_g} -[{id2rel.get(rel_g, rel_g)}]-> {dst_g}")
        return lines

    def _sample_query(rel_id: int) -> dict:
        qs = [q for q in kg.valid_queries if int(q["relation_id"]) == int(rel_id)]
        if not qs:
            qs = [q for q in kg.test_queries if int(q["relation_id"]) == int(rel_id)]
        if not qs:
            raise RuntimeError(f"No queries found for relation_id={rel_id}")
        return random.choice(qs)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1) Risk sample
    q_risk = _sample_query(risk_rel_id)
    head_r = int(q_risk["head_id"])
    cand_r = int(random.choice(q_risk["answers"]))

    node_ids_fix, edge_index_fix, edge_type_fix, g2l_fix = extract_local_subgraph(
        head_r, adjacency, relation2id
    )
    node_ids_dyn, edge_index_dyn, edge_type_dyn, g2l_dyn, meta_dyn = extract_dynamic_subgraph(
        head_r, risk_rel_id, cand_r, adjacency, relation2id
    )

    print("\n[MinimalDynamic][Debug] Risk query")
    print(f"  head_id={head_r}  candidate_risk={cand_r}")
    print(f"  meta={meta_dyn}")
    print(f"  fixed:  nodes={len(node_ids_fix)} edges={edge_index_fix.size(1)}")
    print(f"  dynamic: nodes={len(node_ids_dyn)} edges={edge_index_dyn.size(1)}")
    print("  dynamic edges (global ids):")
    for s in _pretty_edges(edge_index_dyn, edge_type_dyn, g2l_dyn)[:30]:
        print(f"    {s}")

    # 2) Outcome sample
    q_out = _sample_query(outcome_rel_id)
    head_o = int(q_out["head_id"])
    cand_o = int(random.choice(q_out["answers"]))

    node_ids_fix2, edge_index_fix2, edge_type_fix2, g2l_fix2 = extract_local_subgraph(
        head_o, adjacency, relation2id
    )
    node_ids_dyn2, edge_index_dyn2, edge_type_dyn2, g2l_dyn2, meta_dyn2 = extract_dynamic_subgraph(
        head_o, outcome_rel_id, cand_o, adjacency, relation2id
    )

    print("\n[MinimalDynamic][Debug] Outcome query")
    print(f"  head_id={head_o}  candidate_outcome={cand_o}")
    print(f"  meta={meta_dyn2}")
    print(f"  fixed:  nodes={len(node_ids_fix2)} edges={edge_index_fix2.size(1)}")
    print(f"  dynamic: nodes={len(node_ids_dyn2)} edges={edge_index_dyn2.size(1)}")
    print("  dynamic edges (global ids):")
    for s in _pretty_edges(edge_index_dyn2, edge_type_dyn2, g2l_dyn2)[:30]:
        print(f"    {s}")

    # 3) No-valid-path check: pick an obviously invalid tail if possible
    # For risk
    mapping_risk = dynamic_path_support_mapping(head_r, risk_rel_id, adjacency, relation2id)
    invalid_tail_r = None
    for t in range(num_entities):
        if t not in mapping_risk:
            invalid_tail_r = t
            break
    if invalid_tail_r is not None:
        _, edge_index_tmp, _, _, meta_tmp = extract_dynamic_subgraph(
            head_r, risk_rel_id, invalid_tail_r, adjacency, relation2id
        )
        print("\n[MinimalDynamic][Debug] Risk no-path example")
        print(f"  head_id={head_r}  invalid_tail={invalid_tail_r}")
        print(f"  meta={meta_tmp}  edge_count={edge_index_tmp.size(1)}")


def train_minimal_dynamic_subgraph_model(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    print("[MinimalDynamic] train_minimal_dynamic_subgraph_model() called", flush=True)
    kg = load_processed_data(data_root)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    risk_entities, outcome_entities = _load_entity_type_sets(processed)

    print(
        f"[MinimalDynamic] entities={kg.num_entities} relations={kg.num_relations} train_triples={kg.train_triples.shape[0]}",
        flush=True,
    )

    model = MinimalSubgraphModel(
        num_entities=kg.num_entities,
        num_relations=kg.num_relations,
        dim=dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    triples = torch.tensor(kg.train_triples, dtype=torch.long, device=device)
    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    target_mask = (triples[:, 1] == rel_risk) | (triples[:, 1] == rel_outcome)
    target_triples = triples[target_mask]
    if target_triples.size(0) == 0:
        print("[MinimalDynamic] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[MinimalDynamic] Target training triples: {num_target}", flush=True)

    # Cache:
    # - subgraph cache: (h, r, t) -> (node_ids, edge_index, edge_type, g2l, meta)
    # - reachable tail mapping for negatives: (h, r) -> tail_id -> matched_path_types
    subgraph_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int], Dict[str, Any]]] = {}
    reachable_cache: Dict[Tuple[int, int], Dict[int, List[str]]] = {}

    def _get_reachable_map(h_int: int, r_int: int) -> Dict[int, List[str]]:
        key = (h_int, r_int)
        if key in reachable_cache:
            return reachable_cache[key]
        mapping = dynamic_path_support_mapping(h_int, r_int, adjacency, relation2id)
        reachable_cache[key] = mapping
        return mapping

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_target, device=device)
        epoch_loss = 0.0
        used_samples = 0

        for start in range(0, num_target, batch_size):
            idx = perm[start : start + batch_size]
            batch = target_triples[idx]
            h_batch = batch[:, 0]
            r_batch = batch[:, 1]
            t_batch = batch[:, 2]

            batch_loss = 0.0
            effective = 0

            for i in range(h_batch.size(0)):
                h_int = int(h_batch[i].item())
                r_int = int(r_batch[i].item())
                t_int = int(t_batch[i].item())

                # Positive dynamic subgraph
                s_key = (h_int, r_int, t_int)
                if s_key not in subgraph_cache:
                    node_ids, edge_index, edge_type, g2l, meta = extract_dynamic_subgraph(
                        h_int, r_int, t_int, adjacency, relation2id
                    )
                    subgraph_cache[s_key] = (node_ids, edge_index, edge_type, g2l, meta)
                node_ids, edge_index, edge_type, g2l, meta = subgraph_cache[s_key]

                # 若无关键路径支持：先跳过该样本（训练阶段不强行喂 -1e9）
                if not meta.get("has_valid_path", False):
                    continue

                # Negative sampling: sample from other reachable tails for this (h, r)
                reachable_map = _get_reachable_map(h_int, r_int)
                neg_pool = [tail for tail in reachable_map.keys() if int(tail) != int(t_int)]

                # Optional: further restrict by entity type
                if r_int == rel_risk:
                    neg_pool = [t for t in neg_pool if int(t) in set(risk_entities)]
                elif r_int == rel_outcome:
                    neg_pool = [t for t in neg_pool if int(t) in set(outcome_entities)]

                if not neg_pool:
                    continue
                neg_t_int = random.choice(neg_pool)

                s_key_neg = (h_int, r_int, neg_t_int)
                if s_key_neg not in subgraph_cache:
                    node_ids_n, edge_index_n, edge_type_n, g2l_n, meta_n = extract_dynamic_subgraph(
                        h_int, r_int, neg_t_int, adjacency, relation2id
                    )
                    subgraph_cache[s_key_neg] = (node_ids_n, edge_index_n, edge_type_n, g2l_n, meta_n)
                node_ids_n, edge_index_n, edge_type_n, g2l_n, meta_n = subgraph_cache[s_key_neg]

                if not meta_n.get("has_valid_path", False):
                    continue

                # Forward / loss
                node_ids_d = node_ids.to(device)
                edge_index_d = edge_index.to(device)
                edge_type_d = edge_type.to(device)

                node_ids_n_d = node_ids_n.to(device)
                edge_index_n_d = edge_index_n.to(device)
                edge_type_n_d = edge_type_n.to(device)

                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
                tails_t = torch.tensor([t_int], dtype=torch.long, device=device)
                pos_score = model(node_ids_d, edge_index_d, edge_type_d, g2l, heads_t, rels_t, tails_t)
                y_pos = torch.ones_like(pos_score)

                neg_tails_t = torch.tensor([neg_t_int], dtype=torch.long, device=device)
                neg_score = model(node_ids_n_d, edge_index_n_d, edge_type_n_d, g2l_n, heads_t, rels_t, neg_tails_t)
                y_neg = torch.zeros_like(neg_score)

                loss = loss_fn(pos_score, y_pos) + loss_fn(neg_score, y_neg)
                batch_loss += loss
                effective += 1

            if effective > 0:
                batch_loss = batch_loss / effective
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item() * effective
                used_samples += effective

        avg_loss = epoch_loss / used_samples if used_samples > 0 else 0.0
        print(f"[MinimalDynamic] Epoch {epoch}/{epochs} loss={avg_loss:.4f} used={used_samples}", flush=True)

    print("[MinimalDynamic] Training done. Evaluating...", flush=True)

    def score_fn(h_np: np.ndarray, r_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        model.eval()
        n = len(h_np)
        scores = np.full(n, -1e9, dtype=np.float32)

        h_int = int(h_np[0])
        r_int = int(r_np[0])

        reachable_map = _get_reachable_map(h_int, r_int)
        valid_idx: List[int] = [i for i, t in enumerate(t_np) if int(t) in reachable_map]
        if not valid_idx:
            return scores

        valid_tails = [int(t_np[i]) for i in valid_idx]

        with torch.no_grad():
            if STRICT_EVAL_DYNAMIC:
                # Strict (slow): per candidate build dynamic subgraph and score.
                for pos_i, t_global in zip(valid_idx, valid_tails):
                    s_key = (h_int, r_int, int(t_global))
                    if s_key not in subgraph_cache:
                        node_ids_dy, edge_index_dy, edge_type_dy, g2l_dy, meta_dy = extract_dynamic_subgraph(
                            h_int, r_int, int(t_global), adjacency, relation2id
                        )
                        subgraph_cache[s_key] = (node_ids_dy, edge_index_dy, edge_type_dy, g2l_dy, meta_dy)
                    node_ids_dy, edge_index_dy, edge_type_dy, g2l_dy, meta_dy = subgraph_cache[s_key]
                    if not meta_dy.get("has_valid_path", False):
                        continue
                    node_ids_dy = node_ids_dy.to(device)
                    edge_index_dy = edge_index_dy.to(device)
                    edge_type_dy = edge_type_dy.to(device)
                    heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                    rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
                    tails_t = torch.tensor([int(t_global)], dtype=torch.long, device=device)
                    s = model(node_ids_dy, edge_index_dy, edge_type_dy, g2l_dy, heads_t, rels_t, tails_t)
                    scores[pos_i] = float(s.item())
            else:
                # Fast (default): edge-insensitive in current minimal encoder => only need [head] + candidate tails.
                # The dynamic constraint (has_valid_path) is enforced by reachable_map gating.
                unique_nodes = [h_int] + valid_tails
                g2l = {nid: i for i, nid in enumerate(unique_nodes)}
                node_ids_d = torch.tensor(unique_nodes, dtype=torch.long, device=device)
                edge_index_empty = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_type_empty = torch.empty((0,), dtype=torch.long, device=device)

                batch_h = torch.tensor([h_int] * len(valid_tails), dtype=torch.long, device=device)
                batch_r = torch.tensor([r_int] * len(valid_tails), dtype=torch.long, device=device)
                batch_t = torch.tensor(valid_tails, dtype=torch.long, device=device)

                batch_scores = model(
                    node_ids_d, edge_index_empty, edge_type_empty, g2l,
                    batch_h, batch_r, batch_t
                ).cpu().numpy()

                for pos, score in zip(valid_idx, batch_scores):
                    scores[pos] = float(score)
        return scores

    valid_metrics = evaluate_tail_predictions(kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set)
    test_metrics = evaluate_tail_predictions(kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set)
    return {"valid": valid_metrics, "test": test_metrics}


def main() -> None:
    import faulthandler
    import sys

    faulthandler.enable(file=sys.stderr)

    print("[MinimalDynamic] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "subgraph_minimal_dynamic"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MinimalDynamic] Device: {device}", flush=True)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    kg = load_processed_data(data_root)

    # Validation / sanity check prints (required by your spec)
    _print_debug_dynamic_subgraph(
        kg=kg,
        adjacency=adjacency,
        relation2id=relation2id,
        num_entities=kg.num_entities,
        device=device,
    )

    # Train + evaluate
    metrics = train_minimal_dynamic_subgraph_model(data_root, device=device)

    out_path = results_dir / "subgraph_minimal_dynamic_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[MinimalDynamic] Metrics -> {out_path}", flush=True)
    print(f"[MinimalDynamic] valid: {metrics['valid']}", flush=True)
    print(f"[MinimalDynamic] test:  {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()

