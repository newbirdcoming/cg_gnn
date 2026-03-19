from typing import List

import torch
import torch.nn as nn


class LocalRGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        # 保留参数定义接口，后面如果需要可以再打开邻居聚合
        self.num_relations = num_relations
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        极简安全版：当前只做 self-loop 映射，不做按关系的邻居聚合，
        目的是先避免在 Windows + PyTorch 的稀疏乘法里触发底层 BLAS 崩溃。

        这样整个最小模型仍然是“按局部子图选出一小撮节点 -> 查表 + MLP 编码 -> DistMult 打分”，
        先保证训练/评估流程可以完整跑完，后面再迭代增强消息传递。
        """
        return torch.relu(self.self_loop(x))


class LocalRGCNEncoder(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        layers: List[LocalRGCNLayer] = []
        for _ in range(num_layers):
            layers.append(LocalRGCNLayer(dim, dim, num_relations))
        self.layers = nn.ModuleList(layers)

    def forward(
        self, node_ids: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a local subgraph.
        node_ids: (N_sub,) global entity ids
        """
        x = self.entity_emb(node_ids)
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x

