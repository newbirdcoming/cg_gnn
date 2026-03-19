import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import load_processed_data
from .metrics import evaluate_tail_predictions


class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int = 100):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight.data)
        nn.init.xavier_uniform_(self.relation_emb.weight.data)

    def forward(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor):
        h = self.entity_emb(heads)
        r = self.relation_emb(rels)
        t = self.entity_emb(tails)
        return -torch.linalg.norm(h + r - t, dim=-1)  # higher is better


def train_transe(
    data_root: Path,
    dim: int = 64,
    lr: float = 0.001,
    margin: float = 1.0,
    epochs: int = 5,
    batch_size: int = 64,
    eval_every: int = 10,
    device: str = "cpu",
) -> dict:
    kg = load_processed_data(data_root)

    model = TransE(kg.num_entities, kg.num_relations, dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=margin)

    train_triples = torch.tensor(kg.train_triples, dtype=torch.long, device=device)
    num_train = train_triples.shape[0]

    def score_fn(h_np, r_np, t_np):
        model.eval()
        with torch.no_grad():
            h = torch.tensor(h_np, dtype=torch.long, device=device)
            r = torch.tensor(r_np, dtype=torch.long, device=device)
            t = torch.tensor(t_np, dtype=torch.long, device=device)
            scores = model(h, r, t).cpu().numpy()
        return scores

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        epoch_loss = 0.0

        for start in range(0, num_train, batch_size):
            idx = perm[start : start + batch_size]
            batch = train_triples[idx]
            h_pos, r_pos, t_pos = batch[:, 0], batch[:, 1], batch[:, 2]

            # Negative sampling: corrupt tail
            t_neg = torch.randint(
                low=0,
                high=kg.num_entities,
                size=t_pos.shape,
                device=device,
            )

            pos_scores = model(h_pos, r_pos, t_pos)
            neg_scores = model(h_pos, r_pos, t_neg)

            target = torch.ones_like(pos_scores)
            loss = loss_fn(pos_scores, neg_scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= num_train
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"[TransE] epoch {epoch}/{epochs}, loss={epoch_loss:.4f}")

    # Final evaluation on valid/test queries
    print("[TransE] Evaluating on valid/test...")
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
    metrics = train_transe(data_root, device=device)

    out_path = results_dir / "transe_metrics.json"
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[TransE] Metrics written to {out_path}")


if __name__ == "__main__":
    main()

