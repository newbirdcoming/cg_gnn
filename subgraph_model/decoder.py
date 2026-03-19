from typing import Dict

import torch
import torch.nn as nn


class SubgraphDistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(
        self,
        z_local: torch.Tensor,
        global2local: Dict[int, int],
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        DistMult scoring on a local subgraph.
        heads, rels, tails are global ids; heads/tails must map into global2local.
        """
        device = z_local.device
        h_idx = torch.tensor(
            [global2local[int(h.item())] for h in heads], dtype=torch.long, device=device
        )
        t_idx = torch.tensor(
            [global2local[int(t.item())] for t in tails], dtype=torch.long, device=device
        )

        h = z_local[h_idx]
        t = z_local[t_idx]
        r = self.relation_emb(rels.to(device))
        return torch.sum(h * r * t, dim=-1)

