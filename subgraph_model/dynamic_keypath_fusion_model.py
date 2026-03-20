from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.minimal_dynamic_subgraph_model import MinimalSubgraphModel
from subgraph_model.subgraph import build_adjacency
from subgraph_model.subgraph_dynamic import extract_dynamic_subgraph, dynamic_path_support_mapping

# 路径维度顺序（固定，不可改变）：
# P1: 诉求-(包含隐患)->隐患-(导致)->风险
# P2: 诉求-(包含事件)->事件-(触发风险)->风险
# P3: 诉求-(包含实体)->实体-(易感于)->隐患-(导致)->风险
# P4: 诉求-(包含隐患)->隐患-(导致)->风险-(导致)->后果
# P5: 诉求-(包含事件)->事件-(触发风险)->风险-(导致)->后果
PATH_TYPES: List[str] = ["P1", "P2", "P3", "P4", "P5"]
PATH_DIM = len(PATH_TYPES)  # 5
_PATH_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PATH_TYPES)}

# 规则初始权重（固定，不可学习）
# risk(idx=0)：P1/P2 直接两跳全权，P3 三跳减半，P4/P5 与风险任务无关置零
# outcome(idx=1)：P4/P5 直接三跳全权，P1/P2/P3 与后果任务无关置零
_RULE_INIT_WEIGHT: List[List[float]] = [
    [1.0, 1.0, 0.5, 0.0, 0.0],  # risk
    [0.0, 0.0, 0.0, 1.0, 1.0],  # outcome
]


def meta_to_path_feat(meta: Dict[str, Any]) -> torch.Tensor:
    """将 extract_dynamic_subgraph 返回的 meta 转为 (1, 5) 路径特征向量。"""
    vec = [0.0] * PATH_DIM
    for pt in meta.get("matched_path_types", []):
        idx = _PATH_INDEX.get(pt)
        if idx is not None:
            vec[idx] = 1.0
    return torch.tensor([vec], dtype=torch.float32)  # (1, 5)


class DynamicKeypathFusionModel(nn.Module):
    """
    动态子图 graph branch + 规则关键路径 path branch 融合模型。

    score = lambda_r * score_graph + (1 - lambda_r) * score_path

    其中：
    - score_graph  来自 MinimalSubgraphModel（LocalRGCNEncoder + DistMult）
    - score_path   = path_feat @ final_weight[rel_task_idx]
                   final_weight = rule_init_weight + delta_weight
    - lambda_r     = sigmoid(lambda_logit[rel_task_idx])
    - rel_task_idx: 0=包含风险, 1=包含后果
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()

        self.graph_branch = MinimalSubgraphModel(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            num_layers=num_layers,
        )

        # 固定规则先验（不参与梯度）
        self.register_buffer(
            "rule_init_weight",
            torch.tensor(_RULE_INIT_WEIGHT, dtype=torch.float32),  # (2, 5)
        )

        # 可学习修正项，初始化为零保证从纯规则先验出发
        self.delta_weight = nn.Parameter(torch.zeros(2, PATH_DIM))

        # 融合权重 logit，sigmoid 后映射到 (0, 1)；两个目标关系各一个
        self.lambda_logit = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        global2local: Dict[int, int],
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        path_feat: torch.Tensor,  # (B, 5)
        rel_task_idx: int,        # 0=risk, 1=outcome
    ) -> torch.Tensor:
        device = node_ids.device

        score_graph = self.graph_branch(
            node_ids, edge_index, edge_type, global2local, heads, rels, tails
        )  # (B,)

        final_weight = self.rule_init_weight[rel_task_idx] + self.delta_weight[rel_task_idx]  # (5,)
        score_path = (path_feat.to(device) @ final_weight).squeeze(-1)  # (B,) 或 scalar

        lambda_r = torch.sigmoid(self.lambda_logit[rel_task_idx])  # scalar
        return lambda_r * score_graph + (1.0 - lambda_r) * score_path

    def get_fusion_params(self, rel_task_idx: int) -> Dict[str, Any]:
        """返回当前融合参数，供调试打印。"""
        with torch.no_grad():
            fw = (self.rule_init_weight[rel_task_idx] + self.delta_weight[rel_task_idx]).cpu().tolist()
            lr = float(torch.sigmoid(self.lambda_logit[rel_task_idx]).item())
        return {"final_weight": fw, "lambda_r": lr}


# ─────────────────────────────────────────────────────────────
# 数据辅助
# ─────────────────────────────────────────────────────────────

def _load_adjacency(processed: Path) -> Dict[int, List[Tuple[int, int]]]:
    bg_path = processed / "background.txt"
    src_path = bg_path if bg_path.exists() else processed / "train.txt"
    label = "background.txt" if bg_path.exists() else "train.txt (fallback)"
    print(f"[Fusion] Adjacency source: {label}", flush=True)

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
    entity2id = json.loads((processed / "entity2id.json").read_text(encoding="utf-8"))
    risk_entities = [eid for name, eid in entity2id.items() if name.startswith("risk:")]
    outcome_entities = [eid for name, eid in entity2id.items() if name.startswith("outcome:")]
    return risk_entities, outcome_entities


# ─────────────────────────────────────────────────────────────
# 调试打印
# ─────────────────────────────────────────────────────────────

def _print_debug_fusion(
    model: DynamicKeypathFusionModel,
    kg,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
    device: str,
    n_samples: int = 3,
) -> None:
    """对少量样本打印 score_graph / score_path / lambda_r / final_score。"""
    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    id2rel = {v: k for k, v in relation2id.items()}

    model.eval()
    random.seed(0)

    for rel_id, task_idx, tag in [
        (rel_risk, 0, "风险"),
        (rel_outcome, 1, "后果"),
    ]:
        queries = [q for q in kg.valid_queries if int(q["relation_id"]) == rel_id]
        if not queries:
            queries = [q for q in kg.test_queries if int(q["relation_id"]) == rel_id]
        if not queries:
            continue

        printed = 0
        for q in random.sample(queries, min(n_samples, len(queries))):
            h_int = int(q["head_id"])
            for t_int in q["answers"][:1]:
                t_int = int(t_int)
                node_ids, edge_index, edge_type, g2l, meta = extract_dynamic_subgraph(
                    h_int, rel_id, t_int, adjacency, relation2id
                )
                if not meta.get("has_valid_path", False):
                    continue

                pf = meta_to_path_feat(meta).to(device)
                node_ids_d = node_ids.to(device)
                edge_index_d = edge_index.to(device)
                edge_type_d = edge_type.to(device)
                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([rel_id], dtype=torch.long, device=device)
                tails_t = torch.tensor([t_int], dtype=torch.long, device=device)

                with torch.no_grad():
                    sg = model.graph_branch(
                        node_ids_d, edge_index_d, edge_type_d, g2l,
                        heads_t, rels_t, tails_t,
                    )
                    fw = model.rule_init_weight[task_idx] + model.delta_weight[task_idx]
                    sp = float((pf @ fw).item())
                    lr = float(torch.sigmoid(model.lambda_logit[task_idx]).item())
                    fs = lr * float(sg.item()) + (1.0 - lr) * sp

                pts = meta.get("matched_path_types", [])
                print(
                    f"[Fusion][Debug] tag={tag} head={h_int} tail={t_int}\n"
                    f"  path_types={pts}  path_feat={[int(x) for x in pf[0].tolist()]}\n"
                    f"  score_graph={float(sg.item()):.4f}  score_path={sp:.4f}"
                    f"  lambda_r={lr:.4f}  final_score={fs:.4f}",
                    flush=True,
                )
                printed += 1
                if printed >= n_samples:
                    break
            if printed >= n_samples:
                break


# ─────────────────────────────────────────────────────────────
# 训练 + 评估
# ─────────────────────────────────────────────────────────────

def train_dynamic_keypath_fusion_model(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    print("[Fusion] train_dynamic_keypath_fusion_model() called", flush=True)
    kg = load_processed_data(data_root)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    risk_entities, outcome_entities = _load_entity_type_sets(processed)

    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    # rel_id -> task_idx (0=risk, 1=outcome)
    rel_to_task: Dict[int, int] = {rel_risk: 0, rel_outcome: 1}

    print(
        f"[Fusion] entities={kg.num_entities} relations={kg.num_relations}"
        f" train_triples={kg.train_triples.shape[0]}",
        flush=True,
    )

    model = DynamicKeypathFusionModel(
        num_entities=kg.num_entities,
        num_relations=kg.num_relations,
        dim=dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    triples = torch.tensor(kg.train_triples, dtype=torch.long, device=device)
    target_mask = (triples[:, 1] == rel_risk) | (triples[:, 1] == rel_outcome)
    target_triples = triples[target_mask]
    if target_triples.size(0) == 0:
        print("[Fusion] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[Fusion] Target training triples: {num_target}", flush=True)

    # 双缓存：子图缓存 + 路径特征缓存
    SubgraphEntry = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int], Dict[str, Any]]
    subgraph_cache: Dict[Tuple[int, int, int], SubgraphEntry] = {}
    path_feat_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
    reachable_cache: Dict[Tuple[int, int], Dict[int, List[str]]] = {}

    def _get_subgraph(h: int, r: int, t: int) -> SubgraphEntry:
        key = (h, r, t)
        if key not in subgraph_cache:
            subgraph_cache[key] = extract_dynamic_subgraph(h, r, t, adjacency, relation2id)
        return subgraph_cache[key]

    def _get_path_feat(h: int, r: int, t: int, meta: Dict[str, Any]) -> torch.Tensor:
        key = (h, r, t)
        if key not in path_feat_cache:
            path_feat_cache[key] = meta_to_path_feat(meta)
        return path_feat_cache[key]

    def _get_reachable(h: int, r: int) -> Dict[int, List[str]]:
        key = (h, r)
        if key not in reachable_cache:
            reachable_cache[key] = dynamic_path_support_mapping(h, r, adjacency, relation2id)
        return reachable_cache[key]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_target, device=device)
        epoch_loss = 0.0
        used_samples = 0
        skip_pos = 0
        skip_neg = 0

        for start in range(0, num_target, batch_size):
            idx = perm[start: start + batch_size]
            batch = target_triples[idx]

            batch_loss = torch.tensor(0.0, device=device)
            effective = 0

            for i in range(batch.size(0)):
                h_int = int(batch[i, 0].item())
                r_int = int(batch[i, 1].item())
                t_int = int(batch[i, 2].item())
                task_idx = rel_to_task[r_int]

                # ── 正样本子图 ──
                node_ids, edge_index, edge_type, g2l, meta = _get_subgraph(h_int, r_int, t_int)
                if not meta.get("has_valid_path", False):
                    skip_pos += 1
                    continue
                pf_pos = _get_path_feat(h_int, r_int, t_int, meta).to(device)

                # ── 负样本采样 ──
                reachable = _get_reachable(h_int, r_int)
                neg_pool = [t for t in reachable if int(t) != t_int]
                if r_int == rel_risk:
                    neg_pool = [t for t in neg_pool if int(t) in set(risk_entities)]
                elif r_int == rel_outcome:
                    neg_pool = [t for t in neg_pool if int(t) in set(outcome_entities)]
                if not neg_pool:
                    skip_neg += 1
                    continue

                neg_t = random.choice(neg_pool)
                node_ids_n, edge_index_n, edge_type_n, g2l_n, meta_n = _get_subgraph(h_int, r_int, neg_t)
                if not meta_n.get("has_valid_path", False):
                    skip_neg += 1
                    continue
                pf_neg = _get_path_feat(h_int, r_int, neg_t, meta_n).to(device)

                # ── forward ──
                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
                tails_pos_t = torch.tensor([t_int], dtype=torch.long, device=device)
                tails_neg_t = torch.tensor([neg_t], dtype=torch.long, device=device)

                score_pos = model(
                    node_ids.to(device), edge_index.to(device), edge_type.to(device),
                    g2l, heads_t, rels_t, tails_pos_t, pf_pos, task_idx,
                )
                score_neg = model(
                    node_ids_n.to(device), edge_index_n.to(device), edge_type_n.to(device),
                    g2l_n, heads_t, rels_t, tails_neg_t, pf_neg, task_idx,
                )

                loss = loss_fn(score_pos, torch.ones_like(score_pos)) + \
                       loss_fn(score_neg, torch.zeros_like(score_neg))
                batch_loss = batch_loss + loss
                effective += 1

            if effective > 0:
                (batch_loss / effective).backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += batch_loss.item()
                used_samples += effective

        avg_loss = epoch_loss / used_samples if used_samples > 0 else 0.0
        fp = model.get_fusion_params(0)
        print(
            f"[Fusion] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}"
            f"  used={used_samples}  skip_pos={skip_pos}  skip_neg={skip_neg}"
            f"  lambda_risk={fp['lambda_r']:.3f}",
            flush=True,
        )

    print("[Fusion] Training done. Evaluating...", flush=True)

    # ── 评估 ──
    def score_fn(h_np: np.ndarray, r_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        model.eval()
        n = len(h_np)
        scores = np.full(n, -1e9, dtype=np.float32)

        h_int = int(h_np[0])
        r_int = int(r_np[0])
        if r_int not in rel_to_task:
            return scores
        task_idx = rel_to_task[r_int]

        reachable = _get_reachable(h_int, r_int)
        valid_idx = [i for i, t in enumerate(t_np) if int(t) in reachable]
        if not valid_idx:
            return scores

        with torch.no_grad():
            for pos_i in valid_idx:
                t_int = int(t_np[pos_i])
                node_ids, edge_index, edge_type, g2l, meta = _get_subgraph(h_int, r_int, t_int)
                if not meta.get("has_valid_path", False):
                    continue
                pf = _get_path_feat(h_int, r_int, t_int, meta).to(device)
                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
                tails_t = torch.tensor([t_int], dtype=torch.long, device=device)
                s = model(
                    node_ids.to(device), edge_index.to(device), edge_type.to(device),
                    g2l, heads_t, rels_t, tails_t, pf, task_idx,
                )
                scores[pos_i] = float(s.item())

        return scores

    valid_metrics = evaluate_tail_predictions(kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set)
    test_metrics = evaluate_tail_predictions(kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set)
    return {"valid": valid_metrics, "test": test_metrics}


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main() -> None:
    import faulthandler
    import sys
    faulthandler.enable(file=sys.stderr)

    print("[Fusion] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "subgraph_dynamic_fusion"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Fusion] Device: {device}", flush=True)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    kg = load_processed_data(data_root)

    # ── 训练 ──
    metrics = train_dynamic_keypath_fusion_model(data_root, device=device)

    # ── 调试打印（训练后对少量样本输出各分量）──
    model = DynamicKeypathFusionModel(
        num_entities=kg.num_entities,
        num_relations=kg.num_relations,
    ).to(device)
    _print_debug_fusion(model, kg, adjacency, relation2id, device)

    # ── 保存结果 ──
    out_path = results_dir / "subgraph_dynamic_fusion_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Fusion] Metrics -> {out_path}", flush=True)
    print(f"[Fusion] valid: {metrics['valid']}", flush=True)
    print(f"[Fusion] test:  {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()
