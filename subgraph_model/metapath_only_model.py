import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import load_processed_data
from baseline.metrics import evaluate_tail_predictions
from subgraph_model.metapath_subgraph_model import (
    METAPATH_FEAT_MODE,
    _load_adjacency,
    _build_type_sets,
    _build_true_tails_map,
    _extract_metapath_feat_for_tail,
)


class MetapathOnlyModel(nn.Module):
    """
    MetaPath-only 版本：

    - 不再使用 LocalRGCNEncoder，不做局部子图结构嵌入；
    - 仅依赖：
        * 实体基础 embedding e_h, e_t
        * 关系 embedding r
        * 元路径特征经 MLP 编码后的语义向量 z_meta
    - 打分形式：
        score(h, r, t) = DistMult(e_h, r, e_t) + meta_score(z_meta)
      其中 DistMult 部分提供基础图表示能力，meta_score 部分提供规则引导的高阶语义证据。
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 64,
        num_metapaths: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # 元路径编码：5 维特征 -> dim
        self.metapath_encoder = nn.Sequential(
            nn.Linear(num_metapaths, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )
        # 将 z_meta 压缩为一个标量偏置分数
        self.metapath_scorer = nn.Linear(dim, 1)
        nn.init.xavier_uniform_(self.metapath_scorer.weight)

    def forward(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        metapath_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        heads, rels, tails: (B,)
        metapath_feats: (B, num_metapaths)
        """
        device = self.entity_emb.weight.device

        e_h = self.entity_emb(heads.to(device))  # (B, dim)
        e_t = self.entity_emb(tails.to(device))  # (B, dim)
        r = self.relation_emb(rels.to(device))   # (B, dim)

        # 基础 DistMult 分数
        distmult_score = torch.sum(e_h * r * e_t, dim=-1)  # (B,)

        # 元路径语义分数
        z_meta = self.metapath_encoder(metapath_feats.to(device))  # (B, dim)
        meta_score = self.metapath_scorer(z_meta).squeeze(-1)      # (B,)

        return distmult_score + meta_score


def _sample_negative_tail_global(
    h: int,
    r: int,
    t_true: int,
    risk_rel_id: int,
    outcome_rel_id: int,
    risk_entities: List[int],
    outcome_entities: List[int],
    true_tails_map: Dict[Tuple[int, int], set],
) -> int:
    """
    类型约束的全局负采样（不依赖局部子图）：
    - 对“包含风险”，只在全局 risk_entities 中采样；
    - 对“包含后果”，只在全局 outcome_entities 中采样；
    - 避免当前真 tail 和 train 中 (h,r) 的其他真 tail。
    若可用候选集合过小，返回 -1 表示跳过该样本。
    """
    if r == risk_rel_id:
        pool = risk_entities
    elif r == outcome_rel_id:
        pool = outcome_entities
    else:
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


def train_metapath_only_model(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """
    MetaPath-only 训练 + 评估：
    - 不使用局部子图结构嵌入；
    - 仅使用实体/关系基础 embedding + 规则引导的元路径特征。
    """
    print("[MetaOnly] train_metapath_only_model() called", flush=True)
    kg = load_processed_data(data_root)
    print(
        f"[MetaOnly] entities={kg.num_entities}, "
        f"relations={kg.num_relations}, "
        f"train_triples={kg.train_triples.shape[0]}",
        flush=True,
    )

    processed = data_root / "processed"
    entity2id, relation2id, risk_entities, outcome_entities = _build_type_sets(processed)
    adjacency = _load_adjacency(processed)

    model = MetapathOnlyModel(
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
        print("[MetaOnly] WARNING: no target triples, falling back to all", flush=True)
        target_triples = triples

    num_target = target_triples.size(0)
    print(f"[MetaOnly] Target training triples: {num_target}", flush=True)

    true_tails_map = _build_true_tails_map(kg.train_triples)

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

                # 构造正样本元路径特征
                f_pos = _extract_metapath_feat_for_tail(
                    h, t, adjacency, relation2id, METAPATH_FEAT_MODE
                )
                metapath_pos = torch.tensor(
                    [f_pos], dtype=torch.float32, device=device
                )

                heads_t = torch.tensor([h], dtype=torch.long, device=device)
                rels_t = torch.tensor([r], dtype=torch.long, device=device)
                tails_t = torch.tensor([t], dtype=torch.long, device=device)

                pos_score = model(heads_t, rels_t, tails_t, metapath_pos)
                y_pos = torch.ones_like(pos_score)

                # 类型约束的全局负采样
                t_neg = _sample_negative_tail_global(
                    h=h,
                    r=r,
                    t_true=t,
                    risk_rel_id=rel_risk,
                    outcome_rel_id=rel_outcome,
                    risk_entities=risk_entities,
                    outcome_entities=outcome_entities,
                    true_tails_map=true_tails_map,
                )
                if t_neg < 0:
                    continue

                f_neg = _extract_metapath_feat_for_tail(
                    h, t_neg, adjacency, relation2id, METAPATH_FEAT_MODE
                )
                metapath_neg = torch.tensor(
                    [f_neg], dtype=torch.float32, device=device
                )
                neg_tails_t = torch.tensor([t_neg], dtype=torch.long, device=device)

                neg_score = model(heads_t, rels_t, neg_tails_t, metapath_neg)
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
        print(
            f"[MetaOnly] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  used={used_samples}",
            flush=True,
        )

    print("[MetaOnly] Training done. Evaluating...", flush=True)

    def score_fn(h_np: np.ndarray, r_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        """
        批量打分函数：对同一 head / relation 的所有候选 tail 计算得分。
        仍然满足 evaluate_tail_predictions 的接口要求。
        """
        model.eval()
        n = len(h_np)
        scores = np.full(n, -1e9, dtype=np.float32)

        h_int = int(h_np[0])
        r_int = int(r_np[0])

        with torch.no_grad():
            # 直接在背景图 adjacency 上用规则抽元路径特征，不需要 encode 子图
            feats = [
                _extract_metapath_feat_for_tail(
                    h_int, int(t_id), adjacency, relation2id, METAPATH_FEAT_MODE
                )
                for t_id in t_np
            ]
            metapath_batch = torch.tensor(
                feats, dtype=torch.float32, device=device
            )  # (B, num_metapaths)

            batch_h = torch.tensor(
                h_np, dtype=torch.long, device=device
            )  # 重复的 head
            batch_r = torch.tensor(
                r_np, dtype=torch.long, device=device
            )
            batch_t = torch.tensor(
                t_np, dtype=torch.long, device=device
            )

            batch_scores = model(
                batch_h, batch_r, batch_t, metapath_batch
            ).cpu().numpy()
            scores[:] = batch_scores.astype(np.float32)

        return scores

    valid_metrics = evaluate_tail_predictions(
        kg.valid_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    print("[MetaOnly] Valid done.", flush=True)

    test_metrics = evaluate_tail_predictions(
        kg.test_queries, score_fn, kg.num_entities, kg.all_triples_set
    )
    print("[MetaOnly] Test done.", flush=True)

    return {"valid": valid_metrics, "test": test_metrics}


def main() -> None:
    import faulthandler
    import sys

    faulthandler.enable(file=sys.stderr)

    print("[MetaOnly] main() started", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    results_dir = project_root / "results"
    outputs_dir = project_root / "outputs" / "metapath_only"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MetaOnly] Device: {device}", flush=True)

    try:
        metrics = train_metapath_only_model(data_root, device=device)
    except Exception as e:
        import traceback
        print(f"[MetaOnly] FATAL: {e}", flush=True)
        traceback.print_exc()
        return

    out_path = results_dir / "metapath_only_metrics.json"
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_path2 = outputs_dir / "metrics.json"
    out_path2.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[MetaOnly] Metrics -> {out_path}", flush=True)
    print(f"[MetaOnly] Metrics -> {out_path2}", flush=True)
    print(f"[MetaOnly] valid: {metrics['valid']}", flush=True)
    print(f"[MetaOnly] test:  {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()

