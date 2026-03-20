from __future__ import annotations

from typing import Dict, List, Tuple, Any

import torch


def _unique_in_order(items: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def extract_dynamic_subgraph(
    head_id: int,
    relation_id: int,
    tail_id: int,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int], Dict[str, Any]]:
    """
    面向查询三元组 (head_id, relation_id, tail_id) 的动态局部子图构建。

    - 仅保留能支撑以下候选关键路径的节点与边：
      Risk  (诉求, 包含风险, candidate_risk):
        P1: 诉求 -> 隐患 -> 风险
        P2: 诉求 -> 事件 -> 风险
        P3: 诉求 -> 实体 -> 隐患 -> 风险

      Outcome (诉求, 包含后果, candidate_outcome):
        P4: 诉求 -> 隐患 -> 风险 -> 后果
        P5: 诉求 -> 事件 -> 风险 -> 后果

    - complaint(head_id) 与 candidate(tail_id) 必须进入子图；
    - 若无有效关键路径支持：仍返回最小子图 [head_id, tail_id]，但 meta.has_valid_path=False。
    """
    # 任务关系（外层查询关系）
    rel_include_risk = relation2id.get("包含风险")
    rel_include_outcome = relation2id.get("包含后果")

    # 内层路径关系（规则约束）
    rel_include_entity = relation2id.get("包含实体")
    rel_include_hidden = relation2id.get("包含隐患")
    rel_include_event = relation2id.get("包含事件")
    rel_susceptible = relation2id.get("易感于")
    rel_leads_to = relation2id.get("导致")
    rel_trigger_risk = relation2id.get("触发风险")

    task_type: str
    if relation_id == rel_include_risk:
        task_type = "risk"
    elif relation_id == rel_include_outcome:
        task_type = "outcome"
    else:
        # 这份动态子图只为两个任务关系服务；其它关系直接退化为最小子图
        task_type = "unknown"

    nodes: List[int] = [int(head_id), int(tail_id)]
    edges: List[Tuple[int, int, int]] = []  # (src, rel, dst)
    matched_path_types: List[str] = []

    if task_type == "risk":
        # P1: head -(包含隐患)-> hidden -(导致)-> tail
        hidden_candidates = [
            t for r, t in adjacency.get(int(head_id), []) if r == rel_include_hidden
        ]
        for h_node in hidden_candidates:
            for r2, t2 in adjacency.get(int(h_node), []):
                if r2 == rel_leads_to and int(t2) == int(tail_id):
                    nodes.extend([int(h_node), int(tail_id)])
                    edges.append((int(head_id), rel_include_hidden, int(h_node)))
                    edges.append((int(h_node), rel_leads_to, int(tail_id)))
                    matched_path_types.append("P1")

        # P2: head -(包含事件)-> event -(触发风险)-> tail
        event_candidates = [
            t for r, t in adjacency.get(int(head_id), []) if r == rel_include_event
        ]
        for e_node in event_candidates:
            for r2, t2 in adjacency.get(int(e_node), []):
                if r2 == rel_trigger_risk and int(t2) == int(tail_id):
                    nodes.extend([int(e_node), int(tail_id)])
                    edges.append((int(head_id), rel_include_event, int(e_node)))
                    edges.append((int(e_node), rel_trigger_risk, int(tail_id)))
                    matched_path_types.append("P2")

        # P3: head -(包含实体)-> ent -(易感于)-> hidden -(导致)-> tail
        entity_candidates = [
            t for r, t in adjacency.get(int(head_id), []) if r == rel_include_entity
        ]
        for ent_node in entity_candidates:
            for r2, h_node in adjacency.get(int(ent_node), []):
                if r2 != rel_susceptible:
                    continue
                for r3, t3 in adjacency.get(int(h_node), []):
                    if r3 == rel_leads_to and int(t3) == int(tail_id):
                        nodes.extend([int(ent_node), int(h_node), int(tail_id)])
                        edges.append((int(head_id), rel_include_entity, int(ent_node)))
                        edges.append((int(ent_node), rel_susceptible, int(h_node)))
                        edges.append((int(h_node), rel_leads_to, int(tail_id)))
                        matched_path_types.append("P3")

        matched_path_types = sorted(list(set(matched_path_types)))

    elif task_type == "outcome":
        # P4: head -(包含隐患)-> hidden -(导致)-> risk -(导致)-> tail(outcome)
        hidden_candidates = [
            t for r, t in adjacency.get(int(head_id), []) if r == rel_include_hidden
        ]
        for h_node in hidden_candidates:
            for r2, risk_node in adjacency.get(int(h_node), []):
                if r2 != rel_leads_to:
                    continue
                for r3, out_node in adjacency.get(int(risk_node), []):
                    if r3 == rel_leads_to and int(out_node) == int(tail_id):
                        nodes.extend([int(h_node), int(risk_node), int(tail_id)])
                        edges.append((int(head_id), rel_include_hidden, int(h_node)))
                        edges.append((int(h_node), rel_leads_to, int(risk_node)))
                        edges.append((int(risk_node), rel_leads_to, int(tail_id)))
                        matched_path_types.append("P4")

        # P5: head -(包含事件)-> event -(触发风险)-> risk -(导致)-> tail(outcome)
        event_candidates = [
            t for r, t in adjacency.get(int(head_id), []) if r == rel_include_event
        ]
        for e_node in event_candidates:
            for r2, risk_node in adjacency.get(int(e_node), []):
                if r2 != rel_trigger_risk:
                    continue
                for r3, out_node in adjacency.get(int(risk_node), []):
                    if r3 == rel_leads_to and int(out_node) == int(tail_id):
                        nodes.extend([int(e_node), int(risk_node), int(tail_id)])
                        edges.append((int(head_id), rel_include_event, int(e_node)))
                        edges.append((int(e_node), rel_trigger_risk, int(risk_node)))
                        edges.append((int(risk_node), rel_leads_to, int(tail_id)))
                        matched_path_types.append("P5")

        matched_path_types = sorted(list(set(matched_path_types)))

    has_valid_path = len(matched_path_types) > 0

    unique_nodes = _unique_in_order(nodes)
    global2local = {nid: i for i, nid in enumerate(unique_nodes)}

    if not edges:
        # Degenerate case: only [head, tail]
        node_ids = torch.tensor(unique_nodes, dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        meta = {
            "task_type": task_type,
            "has_valid_path": has_valid_path,
            "matched_path_types": matched_path_types,
        }
        return node_ids, edge_index, edge_type, global2local, meta

    src = [global2local[int(h)] for h, _, _ in edges]
    dst = [global2local[int(t)] for _, _, t in edges]
    rel = [int(r) for _, r, _ in edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)
    node_ids = torch.tensor(unique_nodes, dtype=torch.long)

    meta = {
        "task_type": task_type,
        "has_valid_path": has_valid_path,
        "matched_path_types": matched_path_types,
    }
    return node_ids, edge_index, edge_type, global2local, meta


def dynamic_path_support_mapping(
    head_id: int,
    relation_id: int,
    adjacency: Dict[int, List[Tuple[int, int]]],
    relation2id: Dict[str, int],
) -> Dict[int, List[str]]:
    """
    对同一 (head, relation) 预先计算：哪些 candidate tail 存在有效关键路径。

    返回：tail_id -> matched_path_types（去重后）
    """
    rel_include_risk = relation2id.get("包含风险")
    rel_include_outcome = relation2id.get("包含后果")

    rel_include_entity = relation2id.get("包含实体")
    rel_include_hidden = relation2id.get("包含隐患")
    rel_include_event = relation2id.get("包含事件")
    rel_susceptible = relation2id.get("易感于")
    rel_leads_to = relation2id.get("导致")
    rel_trigger_risk = relation2id.get("触发风险")

    h = int(head_id)
    mapping: Dict[int, List[str]] = {}

    def _add(tail: int, path_type: str) -> None:
        tail = int(tail)
        if tail not in mapping:
            mapping[tail] = []
        if path_type not in mapping[tail]:
            mapping[tail].append(path_type)

    if relation_id == rel_include_risk:
        # P1
        for _, hidden in adjacency.get(h, []):
            if _ != rel_include_hidden:
                continue
            for r2, risk_tail in adjacency.get(int(hidden), []):
                if r2 == rel_leads_to:
                    _add(risk_tail, "P1")

        # P2
        for _, event in adjacency.get(h, []):
            if _ != rel_include_event:
                continue
            for r2, risk_tail in adjacency.get(int(event), []):
                if r2 == rel_trigger_risk:
                    _add(risk_tail, "P2")

        # P3
        for _, entity in adjacency.get(h, []):
            if _ != rel_include_entity:
                continue
            for r2, hidden in adjacency.get(int(entity), []):
                if r2 != rel_susceptible:
                    continue
                for r3, risk_tail in adjacency.get(int(hidden), []):
                    if r3 == rel_leads_to:
                        _add(risk_tail, "P3")

    elif relation_id == rel_include_outcome:
        # P4
        for _, hidden in adjacency.get(h, []):
            if _ != rel_include_hidden:
                continue
            for r2, risk_node in adjacency.get(int(hidden), []):
                if r2 != rel_leads_to:
                    continue
                for r3, out_node in adjacency.get(int(risk_node), []):
                    if r3 == rel_leads_to:
                        _add(out_node, "P4")

        # P5
        for _, event in adjacency.get(h, []):
            if _ != rel_include_event:
                continue
            for r2, risk_node in adjacency.get(int(event), []):
                if r2 != rel_trigger_risk:
                    continue
                for r3, out_node in adjacency.get(int(risk_node), []):
                    if r3 == rel_leads_to:
                        _add(out_node, "P5")

    return mapping

