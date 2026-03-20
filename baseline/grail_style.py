"""
GraIL-style Baseline — baseline/grail_style.py

本实现是一个最小可运行、思想对齐的 GraIL-style baseline，
用于当前项目第四章对比实验。

不是对任何外部原始工程的严格逐项复现，仅在以下四个核心设计维度上与 GraIL 思想对齐：
  1. Query-specific 封闭子图（k-hop BFS，无规则过滤）
  2. 纯结构节点特征（dist_h / dist_t / is_head / is_tail，不使用实体 id）
  3. 真实消息传递编码器（GraIL 风格最小 R-GCN，有邻居聚合）
  4. 无规则引导、无路径权重、无 keypath 分支

与主模型（DynamicKeypathFusionModel）的主要区别：
  - 子图：BFS 封闭子图 vs 规则路径过滤子图
  - 节点特征：结构距离特征 vs 全局实体 embedding
  - 编码器：有消息传递 vs self-loop only
  - 打分：MLP(z_h, z_t, r_emb) vs lambda * DistMult + path_score
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.subgraph import build_adjacency

# ─────────────────────────────────────────────────────────────
# 超参
# ─────────────────────────────────────────────────────────────

DEFAULT_K = 2          # BFS 展开跳数（双侧各 k 跳）
NODE_FEAT_DIM = 4      # [dist_h_norm, dist_t_norm, is_head, is_tail]


# ─────────────────────────────────────────────────────────────
# 子图提取
# ─────────────────────────────────────────────────────────────

def _bfs_distances(
    start: int,
    adj_undirected: Dict[int, List[int]],
    k: int,
) -> Dict[int, int]:
    """从 start 出发做 BFS，返回 {node_id: 距离}，最多展开 k 跳。"""
    dist: Dict[int, int] = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        u = queue.popleft()
        if dist[u] >= k:
            continue
        for v in adj_undirected.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _build_undirected_adj(
    adjacency: Dict[int, List[Tuple[int, int]]],
) -> Dict[int, List[int]]:
    """
    将有向邻接表转为无向邻接表（仅保留节点连通性，忽略关系类型）。
    补逆边保证 dist_t 对终端节点（风险/后果）可正确计算。
    """
    undirected: Dict[int, List[int]] = {}
    for u, nbrs in adjacency.items():
        if u not in undirected:
            undirected[u] = []
        for _, v in nbrs:
            undirected[u].append(v)
            if v not in undirected:
                undirected[v] = []
            undirected[v].append(u)
    return undirected


MAX_NODES = 150   # 子图节点上限，防止高枢纽度节点引爆图规模


def extract_grail_subgraph(
    head_id: int,
    relation_id: int,
    tail_id: int,
    adjacency: Dict[int, List[Tuple[int, int]]],
    target_relation_ids: List[int],
    k: int = DEFAULT_K,
    max_nodes: int = MAX_NODES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    GraIL-style query-specific 封闭子图提取。

    策略：
      - 从 head 和 tail 各做 k-hop BFS（基于无向邻接，补逆边）
      - 取两侧邻域的并集作为子图节点
      - 若节点数超过 max_nodes，按 dist_h + dist_t 升序保留最近的节点（head/tail 强制保留）
      - 保留子图节点间的所有原始有向边（排除目标关系类型）
      - 节点特征：[dist_h_norm, dist_t_norm, is_head, is_tail]

    返回：
      node_ids     : (N,)     全局实体 id
      edge_index   : (2, E)   局部边索引
      edge_type    : (E,)     关系 id
      node_feat    : (N, 4)   结构特征
      global2local : Dict[global_id -> local_idx]
    """
    h = int(head_id)
    t = int(tail_id)
    target_rel_set = set(target_relation_ids)

    # ── BFS 距离（无向）──
    undirected = _build_undirected_adj(adjacency)
    dist_h = _bfs_distances(h, undirected, k)
    dist_t = _bfs_distances(t, undirected, k)

    # ── 子图节点：两侧 BFS 邻域的并集，强制含 h/t ──
    node_set = set(dist_h.keys()) | set(dist_t.keys())
    node_set.add(h)
    node_set.add(t)

    # ── max_nodes 修剪：按 dist_h + dist_t 升序，保留最近节点 ──
    if len(node_set) > max_nodes:
        max_d = float(k + 1)
        def _priority(nid: int) -> float:
            return dist_h.get(nid, max_d) + dist_t.get(nid, max_d)
        candidates = sorted(node_set - {h, t}, key=_priority)
        kept = set(candidates[: max_nodes - 2])  # 保留 h/t 后剩余槽位
        node_set = kept | {h, t}

    # 保持确定顺序：h 在首位，t 在第二位，其余按 id 排序
    others = sorted(node_set - {h, t})
    unique_nodes: List[int] = [h, t] + others
    global2local: Dict[int, int] = {nid: i for i, nid in enumerate(unique_nodes)}
    N = len(unique_nodes)

    # ── 子图内有向边（排除目标关系）──
    edges: List[Tuple[int, int, int]] = []
    for u in unique_nodes:
        for rel, v in adjacency.get(u, []):
            if rel in target_rel_set:
                continue
            if v in global2local:
                edges.append((global2local[u], rel, global2local[v]))

    if edges:
        src = [e[0] for e in edges]
        dst = [e[2] for e in edges]
        rel_ids = [e[1] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(rel_ids, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    # ── 节点结构特征 [dist_h_norm, dist_t_norm, is_head, is_tail] ──
    max_dist = float(k + 1)
    feats: List[List[float]] = []
    for nid in unique_nodes:
        dh = float(dist_h.get(nid, k + 1)) / max_dist
        dt = float(dist_t.get(nid, k + 1)) / max_dist
        is_h = 1.0 if nid == h else 0.0
        is_t = 1.0 if nid == t else 0.0
        feats.append([dh, dt, is_h, is_t])

    node_feat = torch.tensor(feats, dtype=torch.float32)   # (N, 4)
    node_ids = torch.tensor(unique_nodes, dtype=torch.long)

    return node_ids, edge_index, edge_type, node_feat, global2local


# ─────────────────────────────────────────────────────────────
# GraIL 风格最小消息传递编码器
# ─────────────────────────────────────────────────────────────

class GraILConvLayer(nn.Module):
    """
    GraIL 风格最小消息传递层：
      对每个节点 v，聚合来自邻居的关系加权消息，再加 self-loop。

      h_v' = ReLU( W_self * h_v + sum_{(u,r,v)} W_r * h_u / deg(v) )
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        # 每种关系独立变换矩阵
        self.rel_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.rel_weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
        edge_type: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        N = x.size(0)
        out = self.self_loop(x)    # (N, out_dim)

        if edge_index.size(1) == 0:
            return torch.relu(out)

        src, dst = edge_index[0], edge_index[1]   # (E,)
        out_dim = out.size(1)

        # 按关系类型逐类聚合，避免 bmm 构造 (E, in, out) 大张量
        # （Windows + PyTorch BLAS 在该维度下会触发 access violation）
        agg = torch.zeros(N, out_dim, device=x.device)
        for r_idx in range(self.num_relations):
            mask = (edge_type == r_idx)
            if not mask.any():
                continue
            src_r = src[mask]
            dst_r = dst[mask]
            msg_r = x[src_r] @ self.rel_weight[r_idx]   # (E_r, out_dim)，2D matmul 安全稳定
            agg.scatter_add_(0, dst_r.unsqueeze(1).expand_as(msg_r), msg_r)

        # 按入度归一化
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1.0).unsqueeze(1)

        out = out + agg / deg
        return torch.relu(out)


class GraILEncoder(nn.Module):
    """
    在当前项目中实现的 GraIL 风格最小消息传递编码器。

    输入：节点结构特征（4 维，不使用实体 id），边类型
    输出：每个节点的隐层表示 z (N, dim)
    """

    def __init__(
        self,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
        node_feat_dim: int = NODE_FEAT_DIM,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, dim)
        self.layers = nn.ModuleList([
            GraILConvLayer(dim, dim, num_relations)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_feat: torch.Tensor,   # (N, 4)
        edge_index: torch.Tensor,  # (2, E)
        edge_type: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        x = torch.relu(self.input_proj(node_feat))
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x   # (N, dim)


# ─────────────────────────────────────────────────────────────
# GraIL-style 模型
# ─────────────────────────────────────────────────────────────

class GraILStyleModel(nn.Module):
    """
    GraIL-style baseline 完整模型。

    打分：score(h, r, t) = MLP( concat(z_h, z_t, r_emb) )
      - z_h, z_t 来自 GraILEncoder（纯结构特征，无全局实体 id）
      - r_emb 来自 relation embedding（relation_id 显式参与打分）
    """

    def __init__(
        self,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraILEncoder(
            num_relations=num_relations,
            dim=dim,
            num_layers=num_layers,
        )
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # MLP: [z_h, z_t, r_emb] (3*dim) -> 1
        self.scorer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        for layer in self.scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        node_feat: torch.Tensor,    # (N, 4)
        edge_index: torch.Tensor,   # (2, E)
        edge_type: torch.Tensor,    # (E,)
        global2local: Dict[int, int],
        heads: torch.Tensor,        # (B,) global ids
        rels: torch.Tensor,         # (B,)
        tails: torch.Tensor,        # (B,)
    ) -> torch.Tensor:              # (B,)
        device = node_feat.device
        z = self.encoder(node_feat, edge_index, edge_type)  # (N, dim)

        h_idx = torch.tensor(
            [global2local[int(h.item())] for h in heads],
            dtype=torch.long, device=device,
        )
        t_idx = torch.tensor(
            [global2local[int(t.item())] for t in tails],
            dtype=torch.long, device=device,
        )
        z_h = z[h_idx]                             # (B, dim)
        z_t = z[t_idx]                             # (B, dim)
        r_emb = self.relation_emb(rels.to(device)) # (B, dim)

        combined = torch.cat([z_h, z_t, r_emb], dim=-1)  # (B, 3*dim)
        return self.scorer(combined).squeeze(-1)           # (B,)


# ─────────────────────────────────────────────────────────────
# 数据辅助
# ─────────────────────────────────────────────────────────────

def _load_adjacency(processed: Path) -> Dict[int, List[Tuple[int, int]]]:
    bg_path = processed / "background.txt"
    src_path = bg_path if bg_path.exists() else processed / "train.txt"
    label = "background.txt" if bg_path.exists() else "train.txt (fallback)"
    print(f"[GraIL] Adjacency source: {label}", flush=True)
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

def _print_debug_grail(
    kg,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
    target_relation_ids: List[int],
    n_samples: int = 2,
) -> None:
    """
    对少量样本打印 GraIL-style query-specific 子图信息：
    节点、边数量、head/tail 角色标记、距离特征示例。
    """
    id2rel = {v: k for k, v in relation2id.items()}
    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")

    random.seed(42)

    for rel_id, tag in [(rel_risk, "风险"), (rel_outcome, "后果")]:
        queries = [q for q in kg.valid_queries if int(q["relation_id"]) == rel_id]
        if not queries:
            queries = [q for q in kg.test_queries if int(q["relation_id"]) == rel_id]
        if not queries:
            continue

        sample_q = random.choice(queries)
        h_int = int(sample_q["head_id"])
        t_int = int(random.choice(sample_q["answers"]))

        node_ids, edge_index, edge_type, node_feat, g2l = extract_grail_subgraph(
            h_int, rel_id, t_int, adjacency, target_relation_ids, k=DEFAULT_K,
        )

        print(f"\n[GraIL][Debug] 任务=包含{tag}  head={h_int}  tail={t_int}", flush=True)
        print(f"  子图节点数={len(node_ids)}  边数={edge_index.size(1)}", flush=True)
        print(f"  节点特征维度={node_feat.shape}", flush=True)

        # 打印前 8 个节点的角色标记与距离特征
        print("  节点角色与距离特征（前8个）:", flush=True)
        for i, nid in enumerate(node_ids[:8].tolist()):
            feat = node_feat[i].tolist()
            role = []
            if feat[2] == 1.0:
                role.append("HEAD")
            if feat[3] == 1.0:
                role.append("TAIL")
            if not role:
                role.append("-")
            print(
                f"    node={nid:5d}  dist_h={feat[0]:.2f}  dist_t={feat[1]:.2f}"
                f"  role={'+'.join(role)}",
                flush=True,
            )

        # 打印前 10 条边
        if edge_index.size(1) > 0:
            local2global = {v: k for k, v in g2l.items()}
            print(f"  边（前10条）:", flush=True)
            for e in range(min(10, edge_index.size(1))):
                s = local2global[int(edge_index[0, e])]
                d = local2global[int(edge_index[1, e])]
                r = int(edge_type[e])
                print(f"    {s} -[{id2rel.get(r, r)}]-> {d}", flush=True)


# ─────────────────────────────────────────────────────────────
# 训练 + 评估
# ─────────────────────────────────────────────────────────────

def train_grail_style_model(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    k: int = DEFAULT_K,
    device: str = "cpu",
) -> dict:
    print("[GraIL] train_grail_style_model() called", flush=True)
    kg = load_processed_data(data_root)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    risk_entities, outcome_entities = _load_entity_type_sets(processed)

    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    target_relation_ids = [rel_risk, rel_outcome]

    print(
        f"[GraIL] entities={kg.num_entities} relations={kg.num_relations}"
        f" train_triples={kg.train_triples.shape[0]}  k={k}",
        flush=True,
    )

    model = GraILStyleModel(
        num_relations=kg.num_relations,
        dim=dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    triples = torch.tensor(kg.train_triples, dtype=torch.long, device=device)
    target_mask = (triples[:, 1] == rel_risk) | (triples[:, 1] == rel_outcome)
    target_triples = triples[target_mask]
    if target_triples.size(0) == 0:
        print("[GraIL] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[GraIL] Target training triples: {num_target}", flush=True)

    # 子图缓存
    SubgraphEntry = Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]
    ]
    subgraph_cache: Dict[Tuple[int, int, int], SubgraphEntry] = {}

    def _get_subgraph(h: int, r: int, t: int) -> SubgraphEntry:
        key = (h, r, t)
        if key not in subgraph_cache:
            subgraph_cache[key] = extract_grail_subgraph(
                h, r, t, adjacency, target_relation_ids, k=k
            )
        return subgraph_cache[key]

    # 预计算 (h, r) -> true tails 集合（用于过滤假负样本）
    true_tails_map: Dict[Tuple[int, int], set] = {}
    for triple in target_triples.cpu().tolist():
        h_i, r_i, t_i = int(triple[0]), int(triple[1]), int(triple[2])
        key = (h_i, r_i)
        if key not in true_tails_map:
            true_tails_map[key] = set()
        true_tails_map[key].add(t_i)

    risk_set = set(risk_entities)
    outcome_set = set(outcome_entities)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_target, device=device)
        epoch_loss = 0.0
        used_samples = 0
        skip_count = 0

        for start in range(0, num_target, batch_size):
            idx = perm[start: start + batch_size]
            batch = target_triples[idx]

            batch_loss = torch.tensor(0.0, device=device)
            effective = 0

            for i in range(batch.size(0)):
                h_int = int(batch[i, 0].item())
                r_int = int(batch[i, 1].item())
                t_int = int(batch[i, 2].item())

                # ── 正样本子图 ──
                node_ids, edge_index, edge_type, node_feat, g2l = _get_subgraph(h_int, r_int, t_int)
                if t_int not in g2l:
                    skip_count += 1
                    continue

                # ── 负样本采样（类型约束 + 过滤真正例）──
                true_tails = true_tails_map.get((h_int, r_int), set())
                if r_int == rel_risk:
                    neg_pool = [e for e in risk_entities if e not in true_tails]
                elif r_int == rel_outcome:
                    neg_pool = [e for e in outcome_entities if e not in true_tails]
                else:
                    neg_pool = []

                if not neg_pool:
                    skip_count += 1
                    continue
                neg_t = random.choice(neg_pool)

                node_ids_n, edge_index_n, edge_type_n, node_feat_n, g2l_n = _get_subgraph(
                    h_int, r_int, neg_t
                )

                # ── forward ──
                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
                tails_pos = torch.tensor([t_int], dtype=torch.long, device=device)
                tails_neg = torch.tensor([neg_t], dtype=torch.long, device=device)

                score_pos = model(
                    node_feat.to(device), edge_index.to(device), edge_type.to(device),
                    g2l, heads_t, rels_t, tails_pos,
                )
                score_neg = model(
                    node_feat_n.to(device), edge_index_n.to(device), edge_type_n.to(device),
                    g2l_n, heads_t, rels_t, tails_neg,
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
        print(
            f"[GraIL] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}"
            f"  used={used_samples}  skipped={skip_count}",
            flush=True,
        )

    print("[GraIL] Training done. Evaluating...", flush=True)

    def score_fn(h_np: np.ndarray, r_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        model.eval()
        n = len(h_np)
        scores = np.full(n, -1e9, dtype=np.float32)

        h_int = int(h_np[0])
        r_int = int(r_np[0])
        if r_int not in (rel_risk, rel_outcome):
            return scores

        with torch.no_grad():
            for i, t_int in enumerate(t_np):
                t_int = int(t_int)
                node_ids, edge_index, edge_type, node_feat, g2l = _get_subgraph(
                    h_int, r_int, t_int
                )
                if t_int not in g2l:
                    continue
                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
                tails_t = torch.tensor([t_int], dtype=torch.long, device=device)
                s = model(
                    node_feat.to(device), edge_index.to(device), edge_type.to(device),
                    g2l, heads_t, rels_t, tails_t,
                )
                scores[i] = float(s.item())

        return scores

    valid_metrics = evaluate_tail_predictions(
        kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    test_metrics = evaluate_tail_predictions(
        kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    return {"valid": valid_metrics, "test": test_metrics}


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main() -> None:
    import faulthandler
    import sys
    faulthandler.enable(file=sys.stderr)

    print("[GraIL] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "grail_style"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GraIL] Device: {device}", flush=True)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    kg = load_processed_data(data_root)

    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    target_relation_ids = [rel_risk, rel_outcome]

    # ── 调试打印 ──
    _print_debug_grail(kg, adjacency, relation2id, target_relation_ids)

    # ── 训练 + 评估 ──
    metrics = train_grail_style_model(data_root, device=device)

    # ── 保存结果 ──
    out_path = results_dir / "grail_style_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[GraIL] Metrics -> {out_path}", flush=True)
    print(f"[GraIL] valid: {metrics['valid']}", flush=True)
    print(f"[GraIL] test:  {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()
