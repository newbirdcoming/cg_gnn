import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.subgraph import build_adjacency, extract_local_subgraph
from subgraph_model.encoder import LocalRGCNEncoder
from subgraph_model.decoder import SubgraphDistMultDecoder


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


def _load_adjacency(processed: Path) -> Dict:
    """加载背景图邻接表（优先用 background.txt，否则退回 train.txt）。"""
    bg_path = processed / "background.txt"
    src_path = bg_path if bg_path.exists() else processed / "train.txt"
    label = "background.txt" if bg_path.exists() else "train.txt (fallback)"
    print(f"[SubgraphModel] Adjacency source: {label}", flush=True)

    rows: List = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            rows.append((int(h), int(r), int(t)))
    return build_adjacency(np.array(rows, dtype=np.int64))


def train_minimal_subgraph_model(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """局部子图模型：训练 + 评估。"""
    print("[SubgraphModel] train_minimal_subgraph_model() called", flush=True)
    kg = load_processed_data(data_root)
    print(
        f"[SubgraphModel] entities={kg.num_entities}, "
        f"relations={kg.num_relations}, "
        f"train_triples={kg.train_triples.shape[0]}",
        flush=True,
    )

    processed = data_root / "processed"
    relation2id = json.loads(
        (processed / "relation2id.json").read_text(encoding="utf-8")
    )
    adjacency = _load_adjacency(processed)

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
        print("[SubgraphModel] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[SubgraphModel] Target training triples: {num_target}", flush=True)

    # ─────────────── 预计算训练 head 的子图缓存 ───────────────
    # 每个 head 只抽一次，epoch 内复用，大幅减少重复计算
    print("[SubgraphModel] Pre-computing subgraphs for training heads...", flush=True)
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
    print(f"[SubgraphModel] Cached {len(subgraph_cache)} subgraphs.", flush=True)

    # ─────────────── 训练循环 ───────────────
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

                    node_ids = node_ids.to(device)
                    edge_index = edge_index.to(device)
                    edge_type = edge_type.to(device)

                    heads_t = torch.tensor([h], dtype=torch.long, device=device)
                    rels_t = torch.tensor([r], dtype=torch.long, device=device)
                    tails_t = torch.tensor([t], dtype=torch.long, device=device)

                    pos_score = model(node_ids, edge_index, edge_type, g2l, heads_t, rels_t, tails_t)
                    y_pos = torch.ones_like(pos_score)

                    # 负样本：从子图节点中选一个不同于 t 的节点
                    local_nodes = node_ids.cpu().tolist()
                    if len(local_nodes) <= 1:
                        continue
                    neg_global = t
                    for candidate in local_nodes:
                        if candidate != t:
                            neg_global = candidate
                            break
                    if neg_global == t:
                        continue

                    neg_tails_t = torch.tensor([neg_global], dtype=torch.long, device=device)
                    neg_score = model(node_ids, edge_index, edge_type, g2l, heads_t, rels_t, neg_tails_t)
                    y_neg = torch.zeros_like(neg_score)

                    loss = loss_fn(pos_score, y_pos) + loss_fn(neg_score, y_neg)
                    batch_loss += loss
                    effective += 1

                except Exception as e:
                    import traceback
                    print(f"[SubgraphModel] sample error h={h} r={r} t={t}: {e}", flush=True)
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
            f"[SubgraphModel] Epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  used={used_samples}",
            flush=True,
        )

    # ─────────────── 评估 ───────────────
    print("[SubgraphModel] Training done. Evaluating...", flush=True)

    def score_fn(h_np: np.ndarray, r_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        """
        批量打分函数。
        evaluate_tail_predictions 每次调用时 h / r 相同，t 是一批候选节点。
        因此只需对该 head 抽一次子图，然后批量打分。
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

            # 子图无效，全部给最低分
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
            batch_h = torch.tensor([h_int] * len(valid_idx), dtype=torch.long, device=device)
            batch_r = torch.tensor([r_int] * len(valid_idx), dtype=torch.long, device=device)
            batch_t = torch.tensor(valid_tails, dtype=torch.long, device=device)

            batch_scores = model(
                node_ids_d, edge_index_d, edge_type_d, g2l,
                batch_h, batch_r, batch_t,
            ).cpu().numpy()

            for pos, score in zip(valid_idx, batch_scores):
                scores[pos] = float(score)

        return scores

    valid_metrics = evaluate_tail_predictions(
        kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    print("[SubgraphModel] Valid done.", flush=True)

    test_metrics = evaluate_tail_predictions(
        kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    print("[SubgraphModel] Test done.", flush=True)

    return {"valid": valid_metrics, "test": test_metrics}


def main():
    import faulthandler, sys
    faulthandler.enable(file=sys.stderr)

    print("[SubgraphModel] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "subgraph"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SubgraphModel] Device: {device}", flush=True)

    try:
        metrics = train_minimal_subgraph_model(data_root, device=device)
    except Exception as e:
        import traceback
        print(f"[SubgraphModel] FATAL: {e}", flush=True)
        traceback.print_exc()
        return

    out_path = results_dir / "subgraph_minimal_metrics.json"
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[SubgraphModel] Metrics -> {out_path}", flush=True)
    print(f"[SubgraphModel] Metrics -> {out_path2}", flush=True)
    print(f"[SubgraphModel] valid: {metrics['valid']}", flush=True)
    print(f"[SubgraphModel] test:  {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()
