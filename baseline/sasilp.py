"""
SASILP-style Baseline — baseline/sasilp.py

在当前项目中实现的最小可运行、思想对齐的 SASILP-style baseline，
用于第四章对比实验（与 GraIL / R-GCN / TransE 做对比）。

相比 GraIL-style baseline，SASILP 扩展了以下 4 个维度：
  1. 子图剪枝：PPR 结构分 + 谐波语义分 的加权节点筛选（替代纯距离剪枝）
  2. 节点初始化：距离标签 + 关系多热语义聚合（替代纯距离标签）
  3. 打分：MLP(z_h, r_emb, z_t, subgraph_mean)（增加子图全局表示）
  4. 消融开关：5 个独立开关，便于论文消融实验

消融开关说明：
  use_structural_score  : 是否用 PPR 计算结构分用于节点筛选
  use_semantic_score    : 是否用谐波接近度计算语义分用于节点筛选
  use_subgraph_pruning  : 是否按 final_score 做节点截断（False = 不截断，使用全部 k-hop 节点）
  use_relation_init     : 是否在节点初始化中加入邻接关系语义聚合
  use_distance_label    : 是否在节点初始化中加入距离标签特征

对于论文细节的处理：
  - PPR 采用稀疏字典形式的幂迭代近似实现（工程近似，非论文精确版本）
  - 语义分采用谐波接近度（节点到 h/t 的调和均值），不依赖学习参数，兼顾速度与可解释性
  - 关系语义初始化采用 relation multi-hot profile @ relation_emb，避免逐节点循环
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.subgraph import build_adjacency

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class SASILPConfig:
    # 模型结构
    dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    # 子图
    k_hop: int = 2
    max_nodes: int = 100
    lambda_score: float = 0.5    # 结构分权重（1-lambda 为语义分权重）
    ppr_alpha: float = 0.15      # PPR 重启概率
    ppr_iters: int = 10          # PPR 幂迭代次数
    # 训练
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 64
    seed: int = 42
    patience: int = 3            # 早停耐心值（验证 MRR 无提升的最大轮数）
    # 消融开关
    use_structural_score: bool = True   # PPR 结构分用于节点筛选
    use_semantic_score: bool = True     # 谐波接近度语义分用于节点筛选
    use_subgraph_pruning: bool = True   # 是否按分数截断节点
    use_relation_init: bool = True      # 节点初始化加入关系语义聚合
    use_distance_label: bool = True     # 节点初始化加入距离标签


# ─────────────────────────────────────────────────────────────
# 图工具
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
    """有向邻接表 → 无向邻接表（补逆边），保证终端节点距离可计算。"""
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


# ─────────────────────────────────────────────────────────────
# PPR（Personalized PageRank）结构分
# ─────────────────────────────────────────────────────────────

def _compute_ppr_local(
    node_list: List[int],
    undirected_adj: Dict[int, List[int]],
    seeds: List[int],
    alpha: float = 0.15,
    n_iter: int = 10,
) -> Dict[int, float]:
    """
    稀疏幂迭代 PPR（工程近似实现）。

    以 seeds（head 和 tail）为重启节点，在候选节点集合上计算
    每个节点的 Personalized PageRank 分数，用于度量节点的结构重要性。

    输入：
      node_list     : 候选节点列表
      undirected_adj: 全图无向邻接表
      seeds         : 重启节点（通常是 [h, t]）
      alpha         : 重启概率
      n_iter        : 幂迭代次数

    返回：
      {node_id: ppr_score}，归一化前的原始分数
    """
    node_set = set(node_list)
    valid_seeds = [s for s in seeds if s in node_set]
    if not valid_seeds:
        return {n: 1.0 / max(len(node_list), 1) for n in node_list}

    # 个性化向量（均匀分配给所有 seed）
    teleport: Dict[int, float] = {n: 0.0 for n in node_list}
    per_seed = 1.0 / len(valid_seeds)
    for s in valid_seeds:
        teleport[s] = per_seed

    # 预计算本地邻居（只看 node_set 内部的边）
    local_adj: Dict[int, List[int]] = {
        u: [v for v in undirected_adj.get(u, []) if v in node_set]
        for u in node_list
    }

    # 幂迭代
    p: Dict[int, float] = dict(teleport)
    for _ in range(n_iter):
        p_new = {n: alpha * teleport.get(n, 0.0) for n in node_list}
        for u in node_list:
            pu = p.get(u, 0.0)
            if pu < 1e-12:
                continue
            nbrs = local_adj[u]
            deg = len(nbrs)
            if deg == 0:
                continue
            contrib = (1.0 - alpha) * pu / deg
            for v in nbrs:
                p_new[v] = p_new.get(v, 0.0) + contrib
        p = p_new

    return p


# ─────────────────────────────────────────────────────────────
# 谐波接近度语义分
# ─────────────────────────────────────────────────────────────

def _compute_harmonic_semantic_score(
    node_list: List[int],
    dist_h: Dict[int, int],
    dist_t: Dict[int, int],
    k: int,
) -> Dict[int, float]:
    """
    谐波接近度语义分（工程近似实现，不依赖学习参数）。

    节点 v 的语义分 = 0.5 * (1/(dist_h+1) + 1/(dist_t+1))
    直觉：越接近 h 和 t 的节点，越可能处于 h→t 的路径上，语义相关性越高。
    """
    scores: Dict[int, float] = {}
    for n in node_list:
        dh = dist_h.get(n, k + 1)
        dt = dist_t.get(n, k + 1)
        scores[n] = 0.5 * (1.0 / (dh + 1) + 1.0 / (dt + 1))
    return scores


# ─────────────────────────────────────────────────────────────
# SASILP 子图提取
# ─────────────────────────────────────────────────────────────

def extract_sasilp_subgraph(
    head_id: int,
    relation_id: int,
    tail_id: int,
    adjacency: Dict[int, List[Tuple[int, int]]],
    target_relation_ids: List[int],
    num_relations: int,
    config: SASILPConfig,
    undirected: Optional[Dict[int, List[int]]] = None,
) -> Tuple[
    torch.Tensor,  # node_ids     (N,)
    torch.Tensor,  # edge_index   (2, E)
    torch.Tensor,  # edge_type    (E,)
    torch.Tensor,  # node_feat    (N, 4)  距离特征
    torch.Tensor,  # rel_profile  (N, R)  关系多热特征
    Dict[int, int],  # global2local
    Dict,            # info
]:
    """
    SASILP query-specific 子图提取。

    流程：
      1. 双侧 k-hop BFS 获得候选节点集合
      2. 计算 structural_score（PPR）和 semantic_score（谐波接近度）
      3. final_score = lambda * struct_norm + (1-lambda) * sem_norm
      4. 按 final_score 保留 top max_nodes 节点（若启用 use_subgraph_pruning）
      5. 构建子图边（排除目标关系）
      6. 为每个节点构造 4 维距离特征 + R 维关系多热特征

    消融开关通过 config 控制，所有开关均影响实际行为。

    返回：
      node_ids    : (N,) 全局 entity id
      edge_index  : (2, E) 局部边索引
      edge_type   : (E,) 关系 id
      node_feat   : (N, 4) 距离特征 [dist_h_norm, dist_t_norm, is_head, is_tail]
      rel_profile : (N, R) 关系多热特征（节点出边关系集合的 one-hot 聚合）
      global2local: Dict[global_id → local_idx]
      info        : 调试信息字典
    """
    h, t = int(head_id), int(tail_id)
    k = config.k_hop
    target_rel_set = set(target_relation_ids)

    if undirected is None:
        undirected = _build_undirected_adj(adjacency)

    # ── 1. BFS 候选集 ──
    dist_h = _bfs_distances(h, undirected, k)
    dist_t = _bfs_distances(t, undirected, k)
    candidate_set = set(dist_h.keys()) | set(dist_t.keys())
    candidate_set.add(h)
    candidate_set.add(t)
    candidate_list = list(candidate_set)

    # ── 2. 节点评分与筛选 ──
    if config.use_subgraph_pruning and len(candidate_list) > config.max_nodes:
        struct_scores: Dict[int, float] = {}
        sem_scores: Dict[int, float] = {}

        if config.use_structural_score:
            ppr_raw = _compute_ppr_local(
                candidate_list, undirected, [h, t], config.ppr_alpha, config.ppr_iters
            )
            max_ppr = max(ppr_raw.values()) or 1.0
            struct_scores = {n: ppr_raw[n] / max_ppr for n in candidate_list}
        else:
            struct_scores = {n: 0.0 for n in candidate_list}

        if config.use_semantic_score:
            sem_raw = _compute_harmonic_semantic_score(candidate_list, dist_h, dist_t, k)
            max_sem = max(sem_raw.values()) or 1.0
            sem_scores = {n: sem_raw[n] / max_sem for n in candidate_list}
        else:
            sem_scores = {n: 0.0 for n in candidate_list}

        lam = config.lambda_score
        if config.use_structural_score and config.use_semantic_score:
            final_scores = {
                n: lam * struct_scores[n] + (1.0 - lam) * sem_scores[n]
                for n in candidate_list
            }
        elif config.use_structural_score:
            final_scores = struct_scores
        elif config.use_semantic_score:
            final_scores = sem_scores
        else:
            # 无评分时退回距离加和（与 GraIL 一致）
            max_d = float(k + 1)
            final_scores = {
                n: 1.0 / (dist_h.get(n, max_d) + dist_t.get(n, max_d) + 1e-9)
                for n in candidate_list
            }

        candidates_sorted = sorted(
            candidate_set - {h, t},
            key=lambda n: -final_scores.get(n, 0.0),
        )
        kept = set(candidates_sorted[: config.max_nodes - 2]) | {h, t}
        node_set = kept
    else:
        node_set = candidate_set

    # ── 3. 有序节点列表 ──
    others = sorted(node_set - {h, t})
    unique_nodes: List[int] = [h, t] + others
    global2local: Dict[int, int] = {nid: i for i, nid in enumerate(unique_nodes)}
    N = len(unique_nodes)

    # ── 4. 子图边（排除目标关系）──
    edges: List[Tuple[int, int, int]] = []
    for u in unique_nodes:
        for rel, v in adjacency.get(u, []):
            if rel not in target_rel_set and v in global2local:
                edges.append((global2local[u], rel, global2local[v]))

    if edges:
        edge_index = torch.tensor([[e[0] for e in edges], [e[2] for e in edges]], dtype=torch.long)
        edge_type = torch.tensor([e[1] for e in edges], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    # ── 5. 距离特征 (N, 4) ──
    max_dist_f = float(k + 1)
    feats: List[List[float]] = []
    for nid in unique_nodes:
        dh = (float(dist_h.get(nid, k + 1)) / max_dist_f) if config.use_distance_label else 0.0
        dt = (float(dist_t.get(nid, k + 1)) / max_dist_f) if config.use_distance_label else 0.0
        feats.append([dh, dt, 1.0 if nid == h else 0.0, 1.0 if nid == t else 0.0])
    node_feat = torch.tensor(feats, dtype=torch.float32)

    # ── 6. 关系多热特征 (N, R) ──
    # 节点出边关系集合的 multi-hot 编码，用于节点语义初始化
    rel_profile = torch.zeros(N, num_relations, dtype=torch.float32)
    if config.use_relation_init:
        for i, nid in enumerate(unique_nodes):
            for r, _ in adjacency.get(nid, []):
                if r < num_relations:
                    rel_profile[i, r] = 1.0

    node_ids = torch.tensor(unique_nodes, dtype=torch.long)

    info = {
        "num_nodes": N,
        "num_edges": int(edge_index.size(1)),
        "num_candidates": len(candidate_set),
    }
    return node_ids, edge_index, edge_type, node_feat, rel_profile, global2local, info


# ─────────────────────────────────────────────────────────────
# SASILP 消息传递层（Windows BLAS 安全）
# ─────────────────────────────────────────────────────────────

class SASILPConvLayer(nn.Module):
    """
    R-GCN 消息传递层（逐关系循环，避免 bmm 大张量触发 Windows BLAS crash）。

    h_v' = ReLU( W_self * h_v + Σ_r Σ_{(u,r,v)∈E} W_r * h_u / deg_in(v) )

    输入：
      x          : (N, in_dim)
      edge_index : (2, E)
      edge_type  : (E,)

    输出：
      (N, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.rel_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.rel_weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        N, out_dim = x.size(0), self.self_loop.out_features
        out = self.self_loop(x)

        if edge_index.size(1) == 0:
            return torch.relu(out)

        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(N, out_dim, device=x.device)

        for r_idx in range(self.num_relations):
            mask = (edge_type == r_idx)
            if not mask.any():
                continue
            src_r, dst_r = src[mask], dst[mask]
            msg_r = x[src_r] @ self.rel_weight[r_idx]  # (E_r, out_dim)
            agg.scatter_add_(0, dst_r.unsqueeze(1).expand_as(msg_r), msg_r)

        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1.0).unsqueeze(1)

        out = out + agg / deg
        return torch.relu(out)


# ─────────────────────────────────────────────────────────────
# SASILP 模型
# ─────────────────────────────────────────────────────────────

class SASILPModel(nn.Module):
    """
    SASILP-style 完整模型。

    节点初始化：
      x_v = input_proj( concat([dist_feat(4), rel_sem_init(dim)]) )
        其中 rel_sem_init(v) = rel_profile(v) @ relation_emb.weight
        若 use_relation_init=False：rel_profile 为全零，等价于仅用距离特征
        若 use_distance_label=False：dist_feat 为全零，等价于仅用关系语义特征

    图编码：多层 SASILPConvLayer

    打分：
      score(h, r, t) = MLP( concat(z_h, r_emb, z_t, subgraph_mean) )
      relation_id 通过 r_emb 显式参与打分

    输入：
      node_feat    : (N, 4)
      rel_profile  : (N, R)
      edge_index   : (2, E)
      edge_type    : (E,)
      global2local : Dict
      heads/rels/tails : (B,) global ids
    """

    def __init__(
        self,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations

        # 关系 embedding（打分 + 节点语义初始化共享）
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # 输入投影：concat([dist_4, rel_sem_dim]) → dim
        self.input_proj = nn.Linear(4 + dim, dim)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # GNN 编码层
        self.conv_layers = nn.ModuleList([
            SASILPConvLayer(dim, dim, num_relations)
            for _ in range(num_layers)
        ])

        # 打分 MLP：concat(z_h, r_emb, z_t, subgraph_mean) → 1
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        for layer in self.scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        node_feat: torch.Tensor,      # (N, 4) 距离特征
        rel_profile: torch.Tensor,    # (N, R) 关系多热特征
        edge_index: torch.Tensor,     # (2, E)
        edge_type: torch.Tensor,      # (E,)
        global2local: Dict[int, int],
        heads: torch.Tensor,          # (B,)
        rels: torch.Tensor,           # (B,)
        tails: torch.Tensor,          # (B,)
    ) -> torch.Tensor:                # (B,)
        device = node_feat.device

        # ── 节点初始化 ──
        # rel_sem_init: (N, dim) = rel_profile @ relation_emb_weight
        rel_sem = rel_profile.to(device) @ self.relation_emb.weight   # (N, dim)
        x = torch.relu(self.input_proj(torch.cat([node_feat.to(device), rel_sem], dim=-1)))

        # ── GNN 编码 ──
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_type)

        # ── 子图全局表示（均值池化）──
        subgraph_mean = x.mean(dim=0, keepdim=True).expand(len(heads), -1)  # (B, dim)

        # ── 取 head / tail 表示 ──
        h_idx = torch.tensor(
            [global2local[int(h.item())] for h in heads], dtype=torch.long, device=device
        )
        t_idx = torch.tensor(
            [global2local[int(t.item())] for t in tails], dtype=torch.long, device=device
        )
        z_h = x[h_idx]
        z_t = x[t_idx]
        r_emb = self.relation_emb(rels.to(device))

        # ── 打分 ──
        combined = torch.cat([z_h, r_emb, z_t, subgraph_mean], dim=-1)  # (B, 4*dim)
        return self.scorer(combined).squeeze(-1)                           # (B,)


# ─────────────────────────────────────────────────────────────
# 数据辅助
# ─────────────────────────────────────────────────────────────

def _load_adjacency(processed: Path) -> Dict[int, List[Tuple[int, int]]]:
    bg_path = processed / "background.txt"
    src_path = bg_path if bg_path.exists() else processed / "train.txt"
    label = "background.txt" if bg_path.exists() else "train.txt (fallback)"
    print(f"[SASILP] Adjacency source: {label}", flush=True)
    rows: List[Tuple[int, int, int]] = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rows.append((int(h), int(r), int(t)))
    import numpy as np
    return build_adjacency(np.array(rows, dtype=np.int64))


def _load_entity_type_sets(processed: Path) -> Tuple[List[int], List[int]]:
    entity2id = json.loads((processed / "entity2id.json").read_text(encoding="utf-8"))
    risk_entities = [eid for name, eid in entity2id.items() if name.startswith("risk:")]
    outcome_entities = [eid for name, eid in entity2id.items() if name.startswith("outcome:")]
    return risk_entities, outcome_entities


# ─────────────────────────────────────────────────────────────
# 调试打印
# ─────────────────────────────────────────────────────────────

def _print_debug_sasilp(
    kg,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
    target_relation_ids: List[int],
    config: SASILPConfig,
    undirected: Dict[int, List[int]],
    n_samples: int = 2,
) -> None:
    """对少量样本打印 SASILP 子图信息与节点特征示例。"""
    id2rel = {v: k for k, v in relation2id.items()}
    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    num_relations = kg.num_relations

    random.seed(42)

    for rel_id, tag in [(rel_risk, "风险"), (rel_outcome, "后果")]:
        queries = [q for q in kg.valid_queries if int(q["relation_id"]) == rel_id]
        if not queries:
            queries = [q for q in kg.test_queries if int(q["relation_id"]) == rel_id]
        if not queries:
            continue

        q = random.choice(queries)
        h_int = int(q["head_id"])
        t_int = int(random.choice(q["answers"]))

        node_ids, edge_index, edge_type, node_feat, rel_profile, g2l, info = extract_sasilp_subgraph(
            h_int, rel_id, t_int, adjacency, target_relation_ids, num_relations, config, undirected
        )

        print(f"\n[SASILP][Debug] 任务=包含{tag}  head={h_int}  tail={t_int}", flush=True)
        print(f"  候选节点数={info['num_candidates']}  子图节点数={info['num_nodes']}"
              f"  边数={info['num_edges']}", flush=True)

        print("  节点特征（前8个）dist_h / dist_t / is_h / is_t / rel_profile_nnz:", flush=True)
        for i, nid in enumerate(node_ids[:8].tolist()):
            feat = node_feat[i].tolist()
            nnz = int(rel_profile[i].sum().item())
            role = "HEAD" if feat[2] == 1.0 else ("TAIL" if feat[3] == 1.0 else "-")
            print(f"    node={nid:5d}  dist_h={feat[0]:.2f}  dist_t={feat[1]:.2f}"
                  f"  role={role:<4}  incident_rels={nnz}", flush=True)

        if edge_index.size(1) > 0:
            local2global = {v: k for k, v in g2l.items()}
            print(f"  边（前8条）:", flush=True)
            for e in range(min(8, edge_index.size(1))):
                s = local2global[int(edge_index[0, e])]
                d = local2global[int(edge_index[1, e])]
                r = int(edge_type[e])
                print(f"    {s} -[{id2rel.get(r, r)}]-> {d}", flush=True)


# ─────────────────────────────────────────────────────────────
# 训练 + 评估
# ─────────────────────────────────────────────────────────────

SubgraphEntry = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, Dict[int, int], Dict
]


def train_sasilp_model(
    data_root: Path,
    config: Optional[SASILPConfig] = None,
    device: str = "cpu",
) -> dict:
    if config is None:
        config = SASILPConfig()

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("[SASILP] train_sasilp_model() called", flush=True)
    print(f"[SASILP] Config: {config}", flush=True)
    kg = load_processed_data(data_root)

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    risk_entities, outcome_entities = _load_entity_type_sets(processed)

    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    target_relation_ids = [rel_risk, rel_outcome]
    risk_set = set(risk_entities)
    outcome_set = set(outcome_entities)

    print(
        f"[SASILP] entities={kg.num_entities} relations={kg.num_relations}"
        f" train_triples={kg.train_triples.shape[0]}  k={config.k_hop}",
        flush=True,
    )

    # 预构建无向邻接（全程复用）
    print("[SASILP] Pre-building undirected adjacency...", flush=True)
    undirected_adj = _build_undirected_adj(adjacency)
    print("[SASILP] Done.", flush=True)

    model = SASILPModel(
        num_relations=kg.num_relations,
        dim=config.dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    triples = torch.tensor(kg.train_triples, dtype=torch.long, device=device)
    target_mask = (triples[:, 1] == rel_risk) | (triples[:, 1] == rel_outcome)
    target_triples = triples[target_mask]
    if target_triples.size(0) == 0:
        print("[SASILP] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[SASILP] Target training triples: {num_target}", flush=True)

    # 子图缓存
    subgraph_cache: Dict[Tuple[int, int, int], SubgraphEntry] = {}

    def _get_subgraph(h: int, r: int, t: int) -> SubgraphEntry:
        key = (h, r, t)
        if key not in subgraph_cache:
            subgraph_cache[key] = extract_sasilp_subgraph(
                h, r, t, adjacency, target_relation_ids,
                kg.num_relations, config, undirected=undirected_adj,
            )
        return subgraph_cache[key]

    # 预计算 (h, r) 的真 tail 集合（过滤假负样本）
    true_tails_map: Dict[Tuple[int, int], set] = {}
    for triple in target_triples.cpu().tolist():
        hi, ri, ti = int(triple[0]), int(triple[1]), int(triple[2])
        key = (hi, ri)
        if key not in true_tails_map:
            true_tails_map[key] = set()
        true_tails_map[key].add(ti)

    # ── 训练循环（含早停）──
    best_valid_mrr = -1.0
    best_model_state = None
    patience_counter = 0

    # 统计子图平均节点数
    total_nodes_logged = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = torch.randperm(num_target, device=device)
        epoch_loss = 0.0
        used_samples = 0
        skip_count = 0

        for start in range(0, num_target, config.batch_size):
            idx = perm[start: start + config.batch_size]
            batch = target_triples[idx]

            batch_loss = torch.tensor(0.0, device=device)
            effective = 0

            for i in range(batch.size(0)):
                h_int = int(batch[i, 0].item())
                r_int = int(batch[i, 1].item())
                t_int = int(batch[i, 2].item())

                # 正样本子图
                node_ids, edge_index, edge_type, node_feat, rel_profile, g2l, info = \
                    _get_subgraph(h_int, r_int, t_int)
                if t_int not in g2l:
                    skip_count += 1
                    continue
                if epoch == 1 and len(total_nodes_logged) < 200:
                    total_nodes_logged.append(info["num_nodes"])

                # 负样本（类型约束 + 过滤真正例）
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

                node_ids_n, edge_index_n, edge_type_n, node_feat_n, rel_profile_n, g2l_n, _ = \
                    _get_subgraph(h_int, r_int, neg_t)

                # forward
                heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
                rels_t = torch.tensor([r_int], dtype=torch.long, device=device)

                score_pos = model(
                    node_feat.to(device), rel_profile.to(device),
                    edge_index.to(device), edge_type.to(device),
                    g2l, heads_t, rels_t,
                    torch.tensor([t_int], dtype=torch.long, device=device),
                )
                score_neg = model(
                    node_feat_n.to(device), rel_profile_n.to(device),
                    edge_index_n.to(device), edge_type_n.to(device),
                    g2l_n, heads_t, rels_t,
                    torch.tensor([neg_t], dtype=torch.long, device=device),
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
        avg_nodes = sum(total_nodes_logged) / len(total_nodes_logged) if total_nodes_logged else 0.0
        print(
            f"[SASILP] Epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}"
            f"  used={used_samples}  skipped={skip_count}"
            + (f"  avg_subgraph_nodes={avg_nodes:.1f}" if epoch == 1 else ""),
            flush=True,
        )

        # ── 验证 + 早停 ──
        print(f"[SASILP] Running validation...", flush=True)
        dist_h_cache: Dict[int, Dict[int, int]] = {}

        def _score_fn_val(h_np, r_np, t_np):
            return _score_fn_impl(h_np, r_np, t_np, model, adjacency, undirected_adj,
                                  target_relation_ids, kg.num_relations, config,
                                  rel_risk, rel_outcome, risk_set, outcome_set,
                                  device, subgraph_cache, dist_h_cache)

        valid_metrics = evaluate_tail_predictions(
            kg.valid_queries, _score_fn_val, kg.num_entities, kg.all_triples_set
        )
        valid_mrr = valid_metrics["mrr"]
        print(f"[SASILP] Valid MRR={valid_mrr:.4f}  hits@1={valid_metrics['hits@1']:.4f}", flush=True)

        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[SASILP] Early stopping at epoch {epoch}", flush=True)
                break

    # ── 加载最优模型 ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[SASILP] Loaded best model (valid MRR={best_valid_mrr:.4f})", flush=True)

    print("[SASILP] Final evaluation...", flush=True)
    dist_h_cache_final: Dict[int, Dict[int, int]] = {}
    eval_count = [0]

    def _score_fn_final(h_np, r_np, t_np):
        eval_count[0] += 1
        if eval_count[0] % 20 == 0:
            print(f"[SASILP] Eval query #{eval_count[0]}...", flush=True)
        return _score_fn_impl(h_np, r_np, t_np, model, adjacency, undirected_adj,
                              target_relation_ids, kg.num_relations, config,
                              rel_risk, rel_outcome, risk_set, outcome_set,
                              device, subgraph_cache, dist_h_cache_final)

    valid_metrics = evaluate_tail_predictions(
        kg.valid_queries, _score_fn_final, kg.num_entities, kg.all_triples_set
    )
    eval_count[0] = 0
    test_metrics = evaluate_tail_predictions(
        kg.test_queries, _score_fn_final, kg.num_entities, kg.all_triples_set
    )

    # 平均子图节点数统计
    avg_nodes_final = sum(total_nodes_logged) / len(total_nodes_logged) if total_nodes_logged else 0.0
    print(f"[SASILP] Avg subgraph nodes (train sample): {avg_nodes_final:.1f}", flush=True)

    return {
        "valid": valid_metrics,
        "test": test_metrics,
        "avg_subgraph_nodes": avg_nodes_final,
    }


def _score_fn_impl(
    h_np, r_np, t_np,
    model, adjacency, undirected_adj,
    target_relation_ids, num_relations, config,
    rel_risk, rel_outcome, risk_set, outcome_set,
    device, subgraph_cache, dist_h_cache,
):
    """
    评估打分函数（内部实现，供 train_sasilp_model 中的 score_fn 复用）。

    关键优化：
      1. 实体类型过滤：只对 risk/outcome 实体打分，其余保持 -1e9
      2. dist_h 按 head 缓存，同一 query 的所有候选复用
    """
    model.eval()
    n = len(h_np)
    scores = np.full(n, -1e9, dtype=np.float32)

    h_int = int(h_np[0])
    r_int = int(r_np[0])
    if r_int not in (rel_risk, rel_outcome):
        return scores

    # 实体类型过滤
    type_set = risk_set if r_int == rel_risk else outcome_set

    # dist_h 缓存
    if h_int not in dist_h_cache:
        dist_h_cache[h_int] = _bfs_distances(h_int, undirected_adj, config.k_hop)
    dist_h = dist_h_cache[h_int]

    k = config.k_hop
    max_dist_f = float(k + 1)
    target_rel_set = set(target_relation_ids)

    with torch.no_grad():
        for i, t_int in enumerate(t_np):
            t_int = int(t_int)
            if t_int not in type_set:
                continue

            # 复用 dist_h，仅对当前 tail 计算 dist_t
            dist_t = _bfs_distances(t_int, undirected_adj, k)
            candidate_set = set(dist_h.keys()) | set(dist_t.keys())
            candidate_set.add(h_int)
            candidate_set.add(t_int)

            # 节点筛选（与 extract_sasilp_subgraph 逻辑一致，避免重复提取全子图）
            if config.use_subgraph_pruning and len(candidate_set) > config.max_nodes:
                candidate_list = list(candidate_set)
                struct_sc: Dict[int, float] = {}
                sem_sc: Dict[int, float] = {}

                if config.use_structural_score:
                    ppr_raw = _compute_ppr_local(
                        candidate_list, undirected_adj, [h_int, t_int],
                        config.ppr_alpha, config.ppr_iters
                    )
                    max_ppr = max(ppr_raw.values()) or 1.0
                    struct_sc = {n: ppr_raw[n] / max_ppr for n in candidate_list}

                if config.use_semantic_score:
                    sem_raw = _compute_harmonic_semantic_score(candidate_list, dist_h, dist_t, k)
                    max_sem = max(sem_raw.values()) or 1.0
                    sem_sc = {n: sem_raw[n] / max_sem for n in candidate_list}

                lam = config.lambda_score
                if config.use_structural_score and config.use_semantic_score:
                    final_scores = {n: lam * struct_sc.get(n, 0.0) + (1-lam) * sem_sc.get(n, 0.0)
                                    for n in candidate_list}
                elif config.use_structural_score:
                    final_scores = struct_sc
                elif config.use_semantic_score:
                    final_scores = sem_sc
                else:
                    final_scores = {n: 1.0 / (dist_h.get(n, k+1) + dist_t.get(n, k+1) + 1e-9)
                                    for n in candidate_list}

                sorted_cands = sorted(candidate_set - {h_int, t_int},
                                      key=lambda n: -final_scores.get(n, 0.0))
                node_set = set(sorted_cands[:config.max_nodes - 2]) | {h_int, t_int}
            else:
                node_set = candidate_set

            others = sorted(node_set - {h_int, t_int})
            unique_nodes = [h_int, t_int] + others
            g2l = {nid: idx for idx, nid in enumerate(unique_nodes)}
            N = len(unique_nodes)

            if t_int not in g2l:
                continue

            # 边
            edges = []
            for u in unique_nodes:
                for rel, v in adjacency.get(u, []):
                    if rel not in target_rel_set and v in g2l:
                        edges.append((g2l[u], rel, g2l[v]))
            if edges:
                ei = torch.tensor([[e[0] for e in edges], [e[2] for e in edges]],
                                   dtype=torch.long, device=device)
                et = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
            else:
                ei = torch.empty((2, 0), dtype=torch.long, device=device)
                et = torch.empty((0,), dtype=torch.long, device=device)

            # 节点特征
            feats = []
            for nid in unique_nodes:
                dh = (float(dist_h.get(nid, k+1)) / max_dist_f) if config.use_distance_label else 0.0
                dt = (float(dist_t.get(nid, k+1)) / max_dist_f) if config.use_distance_label else 0.0
                feats.append([dh, dt, 1.0 if nid == h_int else 0.0, 1.0 if nid == t_int else 0.0])
            nf = torch.tensor(feats, dtype=torch.float32, device=device)

            # 关系多热
            rp = torch.zeros(N, num_relations, dtype=torch.float32, device=device)
            if config.use_relation_init:
                for j, nid in enumerate(unique_nodes):
                    for rel, _ in adjacency.get(nid, []):
                        if rel < num_relations:
                            rp[j, rel] = 1.0

            heads_t = torch.tensor([h_int], dtype=torch.long, device=device)
            rels_t = torch.tensor([r_int], dtype=torch.long, device=device)
            tails_t = torch.tensor([t_int], dtype=torch.long, device=device)
            s = model(nf, rp, ei, et, g2l, heads_t, rels_t, tails_t)
            scores[i] = float(s.item())

    return scores


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main(cfg: Optional[SASILPConfig] = None) -> None:
    import faulthandler
    import sys
    faulthandler.enable(file=sys.stderr)

    print("[SASILP] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "sasilp"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SASILP] Device: {device}", flush=True)

    if cfg is None:
        cfg = SASILPConfig()

    processed = data_root / "processed"
    relation2id = json.loads((processed / "relation2id.json").read_text(encoding="utf-8"))
    adjacency = _load_adjacency(processed)
    kg = load_processed_data(data_root)
    rel_risk = relation2id.get("包含风险")
    rel_outcome = relation2id.get("包含后果")
    target_relation_ids = [rel_risk, rel_outcome]
    undirected_adj = _build_undirected_adj(adjacency)

    # ── 调试打印 ──
    _print_debug_sasilp(kg, adjacency, relation2id, target_relation_ids, cfg, undirected_adj)

    # ── 训练 + 评估 ──
    metrics = train_sasilp_model(data_root, config=cfg, device=device)

    # ── 保存结果 ──
    out_path = results_dir / "sasilp_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SASILP] Metrics -> {out_path}", flush=True)
    print(f"[SASILP] valid: {metrics['valid']}", flush=True)
    print(f"[SASILP] test:  {metrics['test']}", flush=True)
    print(f"[SASILP] avg_subgraph_nodes: {metrics['avg_subgraph_nodes']:.1f}", flush=True)


# ── 消融入口 ──────────────────────────────────────────────────

def main_ablation_no_structural() -> None:
    """消融：关闭 PPR 结构分，只用谐波语义分筛选节点。"""
    cfg = SASILPConfig(use_structural_score=False, use_semantic_score=True)
    main(cfg)


def main_ablation_no_semantic() -> None:
    """消融：关闭语义分，只用 PPR 结构分筛选节点。"""
    cfg = SASILPConfig(use_structural_score=True, use_semantic_score=False)
    main(cfg)


def main_ablation_no_pruning() -> None:
    """消融：关闭节点截断，使用全部 k-hop 候选节点（退化为 GraIL-style 子图）。"""
    cfg = SASILPConfig(use_subgraph_pruning=False)
    main(cfg)


def main_ablation_no_relation_init() -> None:
    """消融：关闭关系语义初始化，只用距离标签（退化为 GraIL-style 节点特征）。"""
    cfg = SASILPConfig(use_relation_init=False)
    main(cfg)


def main_ablation_no_distance() -> None:
    """消融：关闭距离标签，只用关系语义初始化。"""
    cfg = SASILPConfig(use_distance_label=False)
    main(cfg)


if __name__ == "__main__":
    main()
