import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import load_processed_data
from .metrics import evaluate_tail_predictions


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.weight.size(2), device=x.device)

        # message passing per edge: m_j->i = W_r * x_j
        # edge_index: (2, E), edge_type: (E,)
        src, dst = edge_index[0], edge_index[1]

        for r in range(self.num_relations):
            mask = edge_type == r
            if not torch.any(mask):
                continue
            src_r = src[mask]
            dst_r = dst[mask]
            x_src = x[src_r]  # (E_r, in_dim)
            w_r = self.weight[r]  # (in_dim, out_dim)
            msg = x_src @ w_r  # (E_r, out_dim)

            # aggregate to dst node (sum)
            out.index_add_(0, dst_r, msg)

        out = out + self.self_loop(x)
        return torch.relu(out)


class RGCNDistMult(nn.Module):
    """R-GCN encoder + DistMult decoder."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        layers: List[RGCNLayer] = []
        for _ in range(num_layers):
            layers.append(RGCNLayer(dim, dim, num_relations))
        self.layers = nn.ModuleList(layers)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.entity_emb.weight
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x  # final entity embeddings

    def score(
        self,
        x: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """DistMult-style scoring: <h, r, t>."""
        h = x[heads]
        t = x[tails]
        r = self.relation_emb(rels)
        return torch.sum(h * r * t, dim=-1)


def build_edge_index(triples: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # undirected edges to let information flow both ways
    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]

    src = np.concatenate([h, t])
    dst = np.concatenate([t, h])
    et = np.concatenate([r, r])

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_type = torch.tensor(et, dtype=torch.long)
    return edge_index, edge_type


def _build_tail_dict(train_triples: np.ndarray) -> Dict[Tuple[int, int], set]:
    """(head, rel) -> set of true tails in train."""
    mapping: Dict[Tuple[int, int], set] = {}
    for h, r, t in train_triples:
        key = (int(h), int(r))
        if key not in mapping:
            mapping[key] = set()
        mapping[key].add(int(t))
    return mapping


def _build_type_sets(data_root: Path) -> Tuple[Dict[str, int], Dict[str, int], List[int], List[int]]:
    """Load entity/relation mappings and build risk/outcome entity id sets."""
    processed = data_root / "processed"
    entity2id = json.loads((processed / "entity2id.json").read_text(encoding="utf-8"))
    relation2id = json.loads(
        (processed / "relation2id.json").read_text(encoding="utf-8")
    )

    risk_entities = [
        ent_id for name, ent_id in entity2id.items() if name.startswith("risk:")
    ]
    outcome_entities = [
        ent_id for name, ent_id in entity2id.items() if name.startswith("outcome:")
    ]

    return entity2id, relation2id, risk_entities, outcome_entities


def _sample_negative_tails(
    h_pos: torch.Tensor,
    r_pos: torch.Tensor,
    t_pos: torch.Tensor,
    num_entities: int,
    risk_rel_id: int,
    outcome_rel_id: int,
    risk_entities: List[int],
    outcome_entities: List[int],
    true_tails_map: Dict[Tuple[int, int], set],
    device: str,
) -> torch.Tensor:
    """Type-aware negative sampling on tails.

    - 对包含风险，只从风险实体集合采样；
    - 对包含后果，只从后果实体集合采样；
    - 其他关系仍然从全局实体中随机采样。
    - 避免采到当前正例 tail，本身作为硬负样本。
    - 尽量避免采到 train 中 (h,r) 的其他真 tail（利用 true_tails_map）。
    """
    batch_size = h_pos.size(0)
    t_neg = torch.empty_like(t_pos)

    risk_pool = torch.tensor(risk_entities, dtype=torch.long, device=device)
    outcome_pool = torch.tensor(outcome_entities, dtype=torch.long, device=device)

    for i in range(batch_size):
        h = int(h_pos[i].item())
        r = int(r_pos[i].item())
        t_true = int(t_pos[i].item())

        # 选择候选集合
        if r == risk_rel_id and len(risk_entities) > 0:
            pool = risk_pool
        elif r == outcome_rel_id and len(outcome_entities) > 0:
            pool = outcome_pool
        else:
            pool = None

        # 采样直到不是真 tail
        while True:
            if pool is not None:
                idx = torch.randint(
                    low=0, high=pool.size(0), size=(1,), device=device
                )[0]
                candidate = int(pool[idx].item())
            else:
                candidate = int(
                    torch.randint(
                        low=0, high=num_entities, size=(1,), device=device
                    )[0].item()
                )

            if candidate == t_true:
                continue
            if (h, r) in true_tails_map and candidate in true_tails_map[(h, r)]:
                continue
            t_neg[i] = candidate
            break

    return t_neg


def train_rgcn(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    eval_every: int = 10,
    device: str = "cpu",
) -> dict:
    """R-GCN + DistMult baseline with type-aware negative sampling."""
    # 固定随机种子，尽量提高结果可复现性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 可选：略微提高确定性（会有轻微性能影响）
    torch.use_deterministic_algorithms(False)

    print(f"[R-GCN] Using random seed = {seed}, device = {device}")

    kg = load_processed_data(data_root)

    # 图结构只用 train 图构建
    edge_index, edge_type = build_edge_index(kg.train_triples)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # 关系/实体类型信息（风险/后果专用）
    entity2id, relation2id, risk_entities, outcome_entities = _build_type_sets(
        data_root
    )
    risk_rel_id = relation2id.get("包含风险")
    outcome_rel_id = relation2id.get("包含后果")

    true_tails_map = _build_tail_dict(kg.train_triples)

    model = RGCNDistMult(
        kg.num_entities, kg.num_relations, dim=dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    triples = torch.tensor(kg.train_triples, dtype=torch.long, device=device)
    num_train = triples.size(0)

    def score_fn(h_np, r_np, t_np):
        model.eval()
        with torch.no_grad():
            x = model.encode(edge_index, edge_type)
            h = torch.tensor(h_np, dtype=torch.long, device=device)
            r = torch.tensor(r_np, dtype=torch.long, device=device)
            t = torch.tensor(t_np, dtype=torch.long, device=device)
            scores = model.score(x, h, r, t).cpu().numpy()
        return scores

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        epoch_loss = 0.0

        for start in range(0, num_train, batch_size):
            idx = perm[start : start + batch_size]
            batch = triples[idx]
            h_pos, r_pos, t_pos = batch[:, 0], batch[:, 1], batch[:, 2]

            # forward to get embeddings
            x = model.encode(edge_index, edge_type)

            pos_score = model.score(x, h_pos, r_pos, t_pos)

            # negative sampling: tail-only, type-aware for 风险/后果
            t_neg = _sample_negative_tails(
                h_pos=h_pos,
                r_pos=r_pos,
                t_pos=t_pos,
                num_entities=kg.num_entities,
                risk_rel_id=risk_rel_id,
                outcome_rel_id=outcome_rel_id,
                risk_entities=risk_entities,
                outcome_entities=outcome_entities,
                true_tails_map=true_tails_map,
                device=device,
            )
            neg_score = model.score(x, h_pos, r_pos, t_neg)

            y_pos = torch.ones_like(pos_score)
            y_neg = torch.zeros_like(neg_score)

            loss = loss_fn(pos_score, y_pos) + loss_fn(neg_score, y_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= num_train
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"[R-GCN] epoch {epoch}/{epochs}, loss={epoch_loss:.4f}")

    print("[R-GCN] Evaluating on valid/test...")
    valid_metrics = evaluate_tail_predictions(
        kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    test_metrics = evaluate_tail_predictions(
        kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set
    )

    return {"valid": valid_metrics, "test": test_metrics}


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = train_rgcn(data_root, device=device)

    out_path = results_dir / "rgcn_metrics.json"
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[R-GCN] Metrics written to {out_path}")


if __name__ == "__main__":
    main()

