from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def build_adjacency(triples: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build outgoing adjacency list: head_id -> [(rel_id, tail_id), ...].
    """
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for h, r, t in triples:
        adj[int(h)].append((int(r), int(t)))
    return adj


def extract_local_subgraph(
    head_id: int,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    Extract a minimal local subgraph around a complaint node for risk-cause reasoning.

    Kept relations:
      - complaint -> entity:   包含实体
      - complaint -> hidden:   包含隐患
      - complaint -> event:    包含事件
      - entity   -> hidden:    易感于
      - hidden   -> risk:      导致
      - event    -> risk:      触发风险
      - risk     -> outcome:   导致
    """
    rel_include_entity = relation2id.get("包含实体")
    rel_include_hidden = relation2id.get("包含隐患")
    rel_include_event = relation2id.get("包含事件")
    rel_susceptible = relation2id.get("易感于")
    rel_leads_to = relation2id.get("导致")
    rel_trigger_risk = relation2id.get("触发风险")

    nodes: List[int] = [head_id]
    edges: List[Tuple[int, int, int]] = []

    # 1-hop from complaint: 包含实体/隐患/事件
    entity_nodes: List[int] = []
    hidden_nodes: List[int] = []
    event_nodes: List[int] = []

    for r, t in adjacency.get(head_id, []):
        if r == rel_include_entity:
            nodes.append(t)
            edges.append((head_id, r, t))
            entity_nodes.append(t)
        elif r == rel_include_hidden:
            nodes.append(t)
            edges.append((head_id, r, t))
            hidden_nodes.append(t)
        elif r == rel_include_event:
            nodes.append(t)
            edges.append((head_id, r, t))
            event_nodes.append(t)

    # entity -> hidden (易感于)
    for e in entity_nodes:
        for r, t in adjacency.get(e, []):
            if r == rel_susceptible:
                nodes.append(t)
                edges.append((e, r, t))
                hidden_nodes.append(t)

    # hidden -> risk (导致)
    risk_nodes: List[int] = []
    for h in hidden_nodes:
        for r, t in adjacency.get(h, []):
            if r == rel_leads_to:
                nodes.append(t)
                edges.append((h, r, t))
                risk_nodes.append(t)

    # event -> risk (触发风险)
    for e in event_nodes:
        for r, t in adjacency.get(e, []):
            if r == rel_trigger_risk:
                nodes.append(t)
                edges.append((e, r, t))
                risk_nodes.append(t)

    # risk -> outcome (导致)
    for r_node in risk_nodes:
        for r, t in adjacency.get(r_node, []):
            if r == rel_leads_to:
                nodes.append(t)
                edges.append((r_node, r, t))

    # deduplicate nodes while preserving order
    seen = set()
    unique_nodes: List[int] = []
    for nid in nodes:
        if nid not in seen:
            seen.add(nid)
            unique_nodes.append(nid)

    global2local: Dict[int, int] = {nid: i for i, nid in enumerate(unique_nodes)}

    if not edges:
        # Degenerate case: only the head node
        node_ids = torch.tensor(unique_nodes, dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        return node_ids, edge_index, edge_type, global2local

    src = [global2local[h] for h, _, _ in edges]
    dst = [global2local[t] for _, _, t in edges]
    rel = [r for _, r, _ in edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)
    node_ids = torch.tensor(unique_nodes, dtype=torch.long)

    return node_ids, edge_index, edge_type, global2local

