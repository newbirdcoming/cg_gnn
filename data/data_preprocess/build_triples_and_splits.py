import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned_combinnation.xlsx"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 列名
ID_COL = "编号"
ENTITY_COL = "实体"
HIDDEN_COL = "隐患"
RISK_COL = "风险"
EVENT_COL = "事件"
OUTCOME_COL = "后果"


def to_node(label: str, prefix: str) -> str:
    """带类型前缀的节点字符串，避免跨类型同名冲突。"""
    return f"{prefix}:{label}"


def build_triples(df: pd.DataFrame) -> pd.DataFrame:
    """从清洗后的宽表构造三元组 DataFrame。"""
    records: List[Tuple[str, str, str, str]] = []

    for _, row in df.iterrows():
        complaint_id = str(row[ID_COL]).strip()

        def norm_val(raw):
            if pd.isna(raw):
                return None
            s = str(raw).strip()
            return s or None

        entity_val = norm_val(row[ENTITY_COL])
        hidden_val = norm_val(row[HIDDEN_COL])
        risk_val = norm_val(row[RISK_COL])
        event_val = norm_val(row[EVENT_COL])
        outcome_val = norm_val(row[OUTCOME_COL])

        # complaint 节点一定存在
        complaint_node = to_node(complaint_id, "complaint")

        entity_node = to_node(entity_val, "entity") if entity_val else None
        hidden_node = to_node(hidden_val, "hidden") if hidden_val else None
        risk_node = to_node(risk_val, "risk") if risk_val else None
        event_node = to_node(event_val, "event") if event_val else None
        outcome_node = to_node(outcome_val, "outcome") if outcome_val else None

        # 1) 诉求-包含-*（只为存在的字段建边）
        if entity_node:
            records.append((complaint_id, complaint_node, "包含实体", entity_node))
        if hidden_node:
            records.append((complaint_id, complaint_node, "包含隐患", hidden_node))
        if event_node:
            records.append((complaint_id, complaint_node, "包含事件", event_node))
        if risk_node:
            records.append((complaint_id, complaint_node, "包含风险", risk_node))
        if outcome_node:
            records.append((complaint_id, complaint_node, "包含后果", outcome_node))

        # 2) 风险因果链内部边（只在两端都存在时建边）
        if entity_node and hidden_node:
            records.append((complaint_id, entity_node, "易感于", hidden_node))
        if hidden_node and risk_node:
            records.append((complaint_id, hidden_node, "导致", risk_node))
        if event_node and risk_node:
            records.append((complaint_id, event_node, "触发风险", risk_node))
        if risk_node and outcome_node:
            records.append((complaint_id, risk_node, "导致", outcome_node))

    triples_df = pd.DataFrame(
        records, columns=["complaint_id", "head", "relation", "tail"]
    ).drop_duplicates()
    return triples_df


def build_mappings(triples_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """从三元组构建 entity2id / relation2id。"""
    entities = sorted(
        set(triples_df["head"].tolist()) | set(triples_df["tail"].tolist())
    )
    relations = sorted(set(triples_df["relation"].tolist()))

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    return entity2id, relation2id


def split_by_complaint_ids(
    triples_df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """按 complaint_id 划分 train/valid/test（归纳式划分）。"""
    complaint_ids = sorted(triples_df["complaint_id"].unique())

    # 固定随机种子，保证可复现
    rng = pd.Series(complaint_ids).sample(frac=1.0, random_state=42).tolist()
    n = len(rng)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    train_ids = set(rng[:n_train])
    valid_ids = set(rng[n_train : n_train + n_valid])
    test_ids = set(rng[n_train + n_valid :])

    train_df = triples_df[triples_df["complaint_id"].isin(train_ids)].copy()
    valid_df = triples_df[triples_df["complaint_id"].isin(valid_ids)].copy()
    test_df = triples_df[triples_df["complaint_id"].isin(test_ids)].copy()

    return {"train": train_df, "valid": valid_df, "test": test_df}


def triples_to_id_txt(
    split_triples: Dict[str, pd.DataFrame],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> None:
    """将三元组按 id 写入 train/valid/test.txt。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, df in split_triples.items():
        lines: List[str] = []
        for _, row in df.iterrows():
            h = entity2id[row["head"]]
            r = relation2id[row["relation"]]
            t = entity2id[row["tail"]]
            lines.append(f"{h}\t{r}\t{t}")

        out_path = PROCESSED_DIR / f"{split_name}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")


def write_background_graph(
    triples_df: pd.DataFrame,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> None:
    """
    写出一个“背景图”：
    - 合并 train/valid/test 所有 complaint 行
    - 仅剔除目标边：包含风险 / 包含后果
    - 其它边（包含实体/隐患/事件 + 因果链内部边）全部保留

    这个背景图专门服务于局部子图模型，用来在 valid/test 上保证：
    - 诉求节点仍然有包含实体/隐患/事件等输入信息
    - 能沿着实体-隐患-风险-后果链路走到候选风险/后果
    """
    target_rels = {"包含风险", "包含后果"}
    bg_df = triples_df[~triples_df["relation"].isin(target_rels)].copy()

    lines: List[str] = []
    for _, row in bg_df.iterrows():
        h = entity2id[row["head"]]
        r = relation2id[row["relation"]]
        t = entity2id[row["tail"]]
        lines.append(f"{h}\t{r}\t{t}")

    out_path = PROCESSED_DIR / "background.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_queries(
    split_triples: Dict[str, pd.DataFrame],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> None:
    """为 valid/test 构造 queries JSON（只做正例答案，不生成负样本）。"""
    # 反向映射，便于生成 answer_texts
    id2entity = {v: k for k, v in entity2id.items()}

    REL_RISK = "包含风险"
    REL_OUTCOME = "包含后果"
    target_rels = {REL_RISK, REL_OUTCOME}

    for split_name in ["valid", "test"]:
        df = split_triples[split_name]

        # 图中只保留“已观测输入边”：诉求-实体/隐患/事件
        mask_input = df["relation"].isin(["包含实体", "包含隐患", "包含事件"])
        mask_target = df["relation"].isin(target_rels)

        input_df = df[mask_input].copy()
        target_df = df[mask_target].copy()

        # 更新 split_triples 中的图三元组，只保留输入边和因果链内部边
        # 注意：此处假设 df 里除 target 以外的边都可以留下（包括实体-隐患-风险-后果等）
        keep_df = df[~mask_target].copy()
        split_triples[split_name] = keep_df

        # 按 (complaint_id, relation) 聚合答案
        answers: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        answers_text: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for _, row in target_df.iterrows():
            complaint_id = row["complaint_id"]
            relation = row["relation"]
            tail = row["tail"]
            tail_id = entity2id[tail]
            answers[(complaint_id, relation)].append(tail_id)
            answers_text[(complaint_id, relation)].append(tail)

        queries = []
        for (complaint_id, relation), tail_ids in answers.items():
            head_node = to_node(str(complaint_id), "complaint")
            head_id = entity2id[head_node]
            rel_id = relation2id[relation]

            unique_tail_ids = sorted(set(tail_ids))
            unique_texts = sorted(set(answers_text[(complaint_id, relation)]))

            q = {
                "complaint_id": complaint_id,
                "head_id": head_id,
                "relation_id": rel_id,
                "answers": unique_tail_ids,
                "head": head_node,
                "relation": relation,
                "answer_texts": unique_texts,
            }
            queries.append(q)

        out_path = PROCESSED_DIR / f"{split_name}_queries.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)


def main() -> None:
    print(f"读取清洗后的 Excel: {CLEANED_PATH}")
    df = pd.read_excel(CLEANED_PATH, dtype=str)

    triples_df = build_triples(df)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    triples_path = RAW_DIR / "triples.csv"
    triples_df.to_csv(triples_path, index=False, encoding="utf-8")
    print(f"已写出三元组到: {triples_path}，共 {len(triples_df)} 条（含 complaint_id）")

    entity2id, relation2id = build_mappings(triples_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "entity2id.json").write_text(
        json.dumps(entity2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROCESSED_DIR / "relation2id.json").write_text(
        json.dumps(relation2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 entity2id.json / relation2id.json 到: {PROCESSED_DIR}")

    # 按 complaint_id 归纳式划分
    split_frames = split_by_complaint_ids(triples_df)

    # 先基于当前三元组写出 id 格式的 train/valid/test.txt
    # build_queries 会在 valid/test 上剔除目标边并重写 split_triples
    triples_to_id_txt(split_frames, entity2id, relation2id)
    build_queries(split_frames, entity2id, relation2id)

    # 剔除目标边后的图，再次写回 valid/test.txt（覆盖之前的）
    triples_to_id_txt(split_frames, entity2id, relation2id)

    # 另外写出一个“背景图”，供局部子图模型和诊断脚本使用
    write_background_graph(triples_df, entity2id, relation2id)

    print("全部数据集文件构建完成。")


if __name__ == "__main__":
    main()

