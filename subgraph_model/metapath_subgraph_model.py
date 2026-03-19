import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.subgraph import build_adjacency, extract_local_subgraph
from subgraph_model.encoder import LocalRGCNEncoder


# 元路径特征模式：binary / count / log_normalized（默认）
METAPATH_FEAT_MODE = "log_normalized"


class MetapathSubgraphModel(nn.Module):
    """
    局部子图 + 元路径增强 版本：
    - 编码器仍然使用 LocalRGCNEncoder（结构表示）
    - 额外为每个 tail 构造 5 维元路径特征，并映射到与结构同维度
    - 最终 tail 表示：z_t = z_struct_t + z_meta_t
    - 打分仍然使用 DistMult 风格，显式依赖 relation_id
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
        num_metapaths: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = LocalRGCNEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=dim,
            num_layers=num_layers,
        )
        # 关系 embedding（与 DistMult 一致）
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # 元路径编码：5 维特征 -> 两层 MLP -> dim
        # 支持计数 / 归一化后的连续特征
        self.metapath_encoder = nn.Sequential(
            nn.Linear(num_metapaths, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

        # tail 侧门控融合：根据 [z_t_struct, z_meta] 学习逐维门控系数 alpha
        self.metapath_gate = nn.Linear(dim * 2, dim)
        nn.init.xavier_uniform_(self.metapath_gate.weight)

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        global2local: Dict[int, int],
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        metapath_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        metapath_feats: (B, num_metapaths) 针对每个 tail 的元路径特征向量
        """
        device = node_ids.device

        z_local = self.encoder(node_ids, edge_index, edge_type)  # (N_sub, dim)

        # 取 head/tail 的结构表示
        h_idx = torch.tensor(
            [global2local[int(h.item())] for h in heads], dtype=torch.long, device=device
        )
        t_idx = torch.tensor(
            [global2local[int(t.item())] for t in tails], dtype=torch.long, device=device
        )
        z_h = z_local[h_idx]  # (B, dim)
        z_t_struct = z_local[t_idx]  # (B, dim)

        # 元路径特征编码得到语义向量 z_meta
        z_meta = self.metapath_encoder(metapath_feats.to(device))  # (B, dim)

        # 结构 + 语义 门控融合：
        # gate_input = [z_t_struct, z_meta]，alpha 为逐维门控系数
        gate_input = torch.cat([z_t_struct, z_meta], dim=-1)  # (B, 2*dim)
        alpha = torch.sigmoid(self.metapath_gate(gate_input))  # (B, dim)
        z_t = z_t_struct + alpha * z_meta

        # DistMult 打分：<h, r, t_fused>
        r = self.relation_emb(rels.to(device))
        score = torch.sum(z_h * r * z_t, dim=-1)
        return score


def _load_adjacency(processed: Path) -> Dict[int, List[Tuple[int, int]]]:
    """加载背景图邻接表（优先用 background.txt，否则退回 train.txt）。"""
    bg_path = processed / "background.txt"
    src_path = bg_path if bg_path.exists() else processed / "train.txt"
    label = "background.txt" if bg_path.exists() else "train.txt (fallback)"
    print(f"[MetaSubgraph] Adjacency source: {label}", flush=True)

    rows: List[Tuple[int, int, int]] = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rows.append((int(h), int(r), int(t)))
    return build_adjacency(np.array(rows, dtype=np.int64))


def _build_type_sets(processed: Path) -> Tuple[Dict[str, int], Dict[str, int], List[int], List[int]]:
    """加载实体/关系映射，并构建风险/后果实体 id 集合（用于类型约束负采样）。"""
    entity2id = json.loads((processed / "entity2id.json").read_text(encoding="utf-8"))
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))

    risk_entities = [eid for name, eid in entity2id.items() if name.startswith("risk:")]
    outcome_entities = [eid for name, eid in entity2id.items() if name.startswith("outcome:")]

    return entity2id, relation2id, risk_entities, outcome_entities


def _build_true_tails_map(train_triples: np.ndarray) -> Dict[Tuple[int, int], set]:
    """(head, rel) -> train 中所有真 tail 集合，用于避免假负样本。"""
    mapping: Dict[Tuple[int, int], set] = {}
    for h, r, t in train_triples:
        key = (int(h), int(r))
        if key not in mapping:
            mapping[key] = set()
        mapping[key].add(int(t))
    return mapping


def _sample_negative_tail(
    h: int,
    r: int,
    t_true: int,
    risk_rel_id: int,
    outcome_rel_id: int,
    risk_entities: List[int],
    outcome_entities: List[int],
    true_tails_map: Dict[Tuple[int, int], set],
    local_nodes: List[int],
) -> int:
    """
    类型约束 + 局部子图约束的负采样（tail-only）：
    - 对“包含风险”，只在 (local_nodes ∩ 风险节点集合) 中采样；
    - 对“包含后果”，只在 (local_nodes ∩ 后果节点集合) 中采样；
    - 避免当前真 tail 和 (h,r) 下其它真 tail。
    若局部候选集合过小，返回 -1 表示跳过该样本。
    """
    # 构造局部风险/后果候选集
    if r == risk_rel_id:
        pool = [nid for nid in local_nodes if nid in set(risk_entities)]
    elif r == outcome_rel_id:
        pool = [nid for nid in local_nodes if nid in set(outcome_entities)]
    else:
        # 非目标关系，当前最小版可以直接返回 -1（我们只训练两类关系）
        return -1

    if len(pool) <= 1:
        return -1

    true_tails = true_tails_map.get((h, r), set())

    for cand in pool:
        if cand == t_true:
            continue
        if cand in true_tails:
            continue
        return cand

    return -1


def _extract_metapath_feat_for_tail(
    head_id: int,
    tail_id: int,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
    mode: str = METAPATH_FEAT_MODE,
) -> List[float]:
    """
    在背景邻接上、围绕 head 的局部结构，提取某个 tail 节点的 5 维元路径特征。

    支持三种模式：
      - binary:        每种元路径是否存在 (0/1)
      - count:         满足模式的路径实例条数
      - log_normalized:
          先对 count 做 log1p，再按 sum 归一化，得到稳定的连续特征。

    定义的元路径：
      P1: 诉求 -> 隐患 -> 风险
      P2: 诉求 -> 事件 -> 风险
      P3: 诉求 -> 实体 -> 隐患 -> 风险
      P4: 诉求 -> 隐患 -> 风险 -> 后果
      P5: 诉求 -> 事件 -> 风险 -> 后果
    """
    rel_inc_entity = relation2id.get("包含实体")
    rel_inc_hidden = relation2id.get("包含隐患")
    rel_inc_event = relation2id.get("包含事件")
    rel_leads_to = relation2id.get("导致")
    rel_susceptible = relation2id.get("易感于")
    rel_trigger = relation2id.get("触发风险")

    # P1: complaint -(包含隐患)-> hidden -(导致)-> risk
    count_p1 = 0
    for r1, h_node in adjacency.get(head_id, []):
        if r1 != rel_inc_hidden:
            continue
        for r2, r_node in adjacency.get(h_node, []):
            if r2 == rel_leads_to and r_node == tail_id:
                count_p1 += 1

    # P2: complaint -(包含事件)-> event -(触发风险)-> risk
    count_p2 = 0
    for r1, e_node in adjacency.get(head_id, []):
        if r1 != rel_inc_event:
            continue
        for r2, r_node in adjacency.get(e_node, []):
            if r2 == rel_trigger and r_node == tail_id:
                count_p2 += 1

    # P3: complaint -(包含实体)-> entity -(易感于)-> hidden -(导致)-> risk
    count_p3 = 0
    for r1, ent_node in adjacency.get(head_id, []):
        if r1 != rel_inc_entity:
            continue
        for r2, h_node in adjacency.get(ent_node, []):
            if r2 != rel_susceptible:
                continue
            for r3, r_node in adjacency.get(h_node, []):
                if r3 == rel_leads_to and r_node == tail_id:
                    count_p3 += 1

    # P4: complaint -(包含隐患)-> hidden -(导致)-> risk -(导致)-> outcome
    count_p4 = 0
    for r1, h_node in adjacency.get(head_id, []):
        if r1 != rel_inc_hidden:
            continue
        for r2, r_node in adjacency.get(h_node, []):
            if r2 != rel_leads_to:
                continue
            for r3, o_node in adjacency.get(r_node, []):
                if r3 == rel_leads_to and o_node == tail_id:
                    count_p4 += 1

    # P5: complaint -(包含事件)-> event -(触发风险)-> risk -(导致)-> outcome
    count_p5 = 0
    for r1, e_node in adjacency.get(head_id, []):
        if r1 != rel_inc_event:
            continue
        for r2, r_node in adjacency.get(e_node, []):
            if r2 != rel_trigger:
                continue
            for r3, o_node in adjacency.get(r_node, []):
                if r3 == rel_leads_to and o_node == tail_id:
                    count_p5 += 1

    counts = [float(count_p1), float(count_p2), float(count_p3), float(count_p4), float(count_p5)]

    mode = (mode or "log_normalized").lower()
    if mode == "binary":
        return [1.0 if c > 0.0 else 0.0 for c in counts]
    if mode == "count":
        return counts

    # 默认：log_normalized
    feats = [math.log1p(c) for c in counts]
    s = sum(feats)
    if s > 0.0:
        feats = [v / s for v in feats]
    return feats


def train_metapath_subgraph_model(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """
    局部子图 + 元路径增强 训练 + 评估（最小可运行版）。
    """
    print("[MetaSubgraph] train_metapath_subgraph_model() called", flush=True)
    kg = load_processed_data(data_root)
    print(
        f"[MetaSubgraph] entities={kg.num_entities}, "
        f"relations={kg.num_relations}, "
        f"train_triples={kg.train_triples.shape[0]}",
        flush=True,
    )

    processed = data_root / "processed"
    entity2id, relation2id, risk_entities, outcome_entities = _build_type_sets(processed)
    adjacency = _load_adjacency(processed)

    model = MetapathSubgraphModel(
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
        print("[MetaSubgraph] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[MetaSubgraph] Target training triples: {num_target}", flush=True)

    # true tails map for avoiding false negatives
    true_tails_map = _build_true_tails_map(kg.train_triples)

    # 预计算 head 的子图缓存
    print("[MetaSubgraph] Pre-computing subgraphs for training heads...", flush=True)
    subgraph_cache: Dict[int, tuple] = {}
    unique_heads = target_triples[:, 0].unique().cpu().tolist()
    for h_int in unique_heads:
        h_int = int(h_int)
        node_ids_c, edge_index_c, edge_type_c, g2l_c = extract_local_subgraph(
            h_int, adjacency, relation2id
        )
        subgraph_cache[h_int] = (
            node_ids_c.to(device),
            edge_index_c.to(device),
            edge_type_c.to(device),
            g2l_c,
        )
    print(f"[MetaSubgraph] Cached {len(subgraph_cache)} subgraphs.", flush=True)

    # 训练循环
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
                h = int(h_batch[i].item())
                r = int(r_batch[i].item())
                t = int(t_batch[i].item())

                try:
                    node_ids, edge_index, edge_type, g2l = subgraph_cache[h]

                    if len(node_ids) <= 1 or edge_index.size(1) == 0 or t not in g2l:
                        continue

                    local_nodes = node_ids.cpu().tolist()

                    # 正样本元路径特征（计数/归一化后的 5 维向量）
                    f_pos = _extract_metapath_feat_for_tail(
                        h, t, adjacency, relation2id, METAPATH_FEAT_MODE
                    )
                    metapath_pos = torch.tensor(
                        [f_pos], dtype=torch.float32, device=device
                    )

                    heads_t = torch.tensor([h], dtype=torch.long, device=device)
                    rels_t = torch.tensor([r], dtype=torch.long, device=device)
                    tails_t = torch.tensor([t], dtype=torch.long, device=device)

                    pos_score = model(
                        node_ids, edge_index, edge_type, g2l,
                        heads_t, rels_t, tails_t, metapath_pos,
                    )
                    y_pos = torch.ones_like(pos_score)

                    # 类型约束 + 局部子图约束的负采样
                    t_neg = _sample_negative_tail(
                        h=h,
                        r=r,
                        t_true=t,
                        risk_rel_id=rel_risk,
                        outcome_rel_id=rel_outcome,
                        risk_entities=risk_entities,
                        outcome_entities=outcome_entities,
                        true_tails_map=true_tails_map,
                        local_nodes=local_nodes,
                    )
                    if t_neg < 0:
                        continue

                    f_neg = _extract_metapath_feat_for_tail(
                        h, t_neg, adjacency, relation2id, METAPATH_FEAT_MODE
                    )
                    metapath_neg = torch.tensor(
                        [f_neg], dtype=torch.float32, device=device
                    )
                    neg_tails_t = torch.tensor(
                        [t_neg], dtype=torch.long, device=device
                    )

                    neg_score = model(
                        node_ids, edge_index, edge_type, g2l,
                        heads_t, rels_t, neg_tails_t, metapath_neg,
                    )
                    y_neg = torch.zeros_like(neg_score)

                    loss = loss_fn(pos_score, y_pos) + loss_fn(neg_score, y_neg)
                    batch_loss += loss
                    effective += 1

                except Exception as e:
                    import traceback
                    print(f"[MetaSubgraph] sample error h={h} r={r} t={t}: {e}", flush=True)
                    traceback.print_exc()
                    continue

            if effective > 0:
                batch_loss = batch_loss / effective
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item() * effective
                used_samples += effective

        avg_loss = epoch_loss / used_samples if used_samples > 0 else 0.0
        print(
            f"[MetaSubgraph] Epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  used={used_samples}",
            flush=True,
        )

    # 评估
    print("[MetaSubgraph] Training done. Evaluating...", flush=True)

    def score_fn(h_np: np.ndarray, r_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        """
        批量打分函数：对同一 head / relation 的所有候选 tail 计算得分。
        """
        model.eval()
        n = len(h_np)
        scores = np.full(n, -1e9, dtype=np.float32)

        h_int = int(h_np[0])
        r_int = int(r_np[0])

        with torch.no_grad():
            node_ids, edge_index, edge_type, g2l = extract_local_subgraph(
                h_int, adjacency, relation2id
            )
            if len(node_ids) <= 1 or edge_index.size(1) == 0:
                return scores

            node_ids_d = node_ids.to(device)
            edge_index_d = edge_index.to(device)
            edge_type_d = edge_type.to(device)

            # 找出哪些候选 tail 在子图里
            valid_idx = [i for i, t in enumerate(t_np) if int(t) in g2l]
            if not valid_idx:
                return scores

            valid_tails = [int(t_np[i]) for i in valid_idx]

            # 构造元路径特征（对所有候选 tail 一次性计算）
            feats = [
                _extract_metapath_feat_for_tail(
                    h_int, t_id, adjacency, relation2id, METAPATH_FEAT_MODE
                )
                for t_id in valid_tails
            ]
            metapath_batch = torch.tensor(
                feats, dtype=torch.float32, device=device
            )  # (B, num_metapaths)

            batch_h = torch.tensor(
                [h_int] * len(valid_idx), dtype=torch.long, device=device
            )
            batch_r = torch.tensor(
                [r_int] * len(valid_idx), dtype=torch.long, device=device
            )
            batch_t = torch.tensor(
                valid_tails, dtype=torch.long, device=device
            )

            batch_scores = model(
                node_ids_d, edge_index_d, edge_type_d, g2l,
                batch_h, batch_r, batch_t, metapath_batch,
            ).cpu().numpy()

            for pos, score in zip(valid_idx, batch_scores):
                scores[pos] = float(score)

        return scores

    valid_metrics = evaluate_tail_predictions(
        kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    print("[MetaSubgraph] Valid done.", flush=True)

    test_metrics = evaluate_tail_predictions(
        kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    print("[MetaSubgraph] Test done.", flush=True)

    return {"valid": valid_metrics, "test": test_metrics}


def main() -> None:
    import faulthandler
    import sys

    faulthandler.enable(file=sys.stderr)

    print("[MetaSubgraph] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "subgraph_metapath"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MetaSubgraph] Device: {device}", flush=True)

    try:
        metrics = train_metapath_subgraph_model(data_root, device=device)
    except Exception as e:
        import traceback
        print(f"[MetaSubgraph] FATAL: {e}", flush=True)
        traceback.print_exc()
        return

    # 写结果：results/ + outputs/subgraph_metapath/
    out_path = results_dir / "subgraph_metapath_metrics.json"
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[MetaSubgraph] Metrics -> {out_path}", flush=True)
    print(f"[MetaSubgraph] Metrics -> {out_path2}", flush=True)
    print(f"[MetaSubgraph] valid: {metrics['valid']}", flush=True)
    print(f"[MetaSubgraph] test:  {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()

