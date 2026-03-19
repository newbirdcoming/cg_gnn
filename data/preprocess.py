import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TRIPLES_PATH = PROJECT_ROOT / "data" / "raw" / "triples.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALLOWED_RELATIONS = {
    "包含实体",
    "包含隐患",
    "包含事件",
    "包含风险",
    "包含后果",
    "易感于",
    "导致",
    "触发风险",
}

TARGET_RELATIONS = {"包含风险", "包含后果"}
INPUT_RELATIONS = {"包含实体", "包含隐患", "包含事件"}


def load_and_clean_triples() -> pd.DataFrame:
    """从 data/raw/triples.csv 读取并清洗三元组."""
    print(f"读取原始三元组: {RAW_TRIPLES_PATH}")
    df = pd.read_csv(RAW_TRIPLES_PATH, dtype=str)

    required_cols = {"complaint_id", "head", "relation", "tail"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"triples.csv 缺少列: {missing}")

    # 去掉首尾空格
    for col in ["complaint_id", "head", "relation", "tail"]:
        df[col] = df[col].astype(str).str.strip()

    # 丢弃关键字段为空的行
    before = len(df)
    df = df[
        (df["complaint_id"] != "")
        & (df["head"] != "")
        & (df["relation"] != "")
        & (df["tail"] != "")
    ]
    print(f"丢弃关键字段为空的行: {before - len(df)} 行")

    # 只保留允许的关系
    before = len(df)
    df = df[df["relation"].isin(ALLOWED_RELATIONS)].copy()
    print(f"丢弃不在允许关系集合内的行: {before - len(df)} 行")

    # 去重
    before = len(df)
    df = df.drop_duplicates()
    print(f"删除重复三元组: {before - len(df)} 行")

    print(f"清洗后三元组数: {len(df)}，诉求数: {df['complaint_id'].nunique()}")
    return df


def build_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """从三元组构建 entity2id / relation2id."""
    entities = sorted(set(df["head"].tolist()) | set(df["tail"].tolist()))
    relations = sorted(set(df["relation"].tolist()))

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    print(f"实体数: {len(entities)}，关系数: {len(relations)}")
    return entity2id, relation2id


def split_complaints(
    df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.1
) -> Dict[str, List[str]]:
    """按 complaint_id 做 7:1:2 归纳式切分."""
    complaint_ids = sorted(df["complaint_id"].unique())
    n = len(complaint_ids)

    # 打乱，固定随机种子保证可复现
    shuffled = (
        pd.Series(complaint_ids).sample(frac=1.0, random_state=42).tolist()
    )

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    train_ids = shuffled[:n_train]
    valid_ids = shuffled[n_train : n_train + n_valid]
    test_ids = shuffled[n_train + n_valid :]

    print(
        f"complaint 划分: train={len(train_ids)}, "
        f"valid={len(valid_ids)}, test={len(test_ids)}"
    )

    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


def write_complaint_split(split_ids: Dict[str, List[str]]) -> None:
    out_path = PROCESSED_DIR / "complaint_split.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 complaint_split.json: {out_path}")


def triples_to_graph_and_queries(
    df: pd.DataFrame,
    split_ids: Dict[str, List[str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> None:
    """根据切分写出 train/valid/test 图和 valid/test 的 queries."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 先处理 train：保留所有边
    train_df = df[df["complaint_id"].isin(split_ids["train"])].copy()
    _write_id_triples(train_df, entity2id, relation2id, "train")

    # valid/test 需要剔除目标边，并构造 queries
    for split in ["valid", "test"]:
        split_df = df[df["complaint_id"].isin(split_ids[split])].copy()

        mask_target = split_df["relation"].isin(TARGET_RELATIONS)
        target_df = split_df[mask_target].copy()
        graph_df = split_df[~mask_target].copy()

        # 写图（不包含目标边）
        _write_id_triples(graph_df, entity2id, relation2id, split)

        # 构造 queries
        _write_queries(target_df, entity2id, relation2id, split)


def _write_id_triples(
    df: pd.DataFrame,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    split_name: str,
) -> None:
    lines: List[str] = []
    for _, row in df.iterrows():
        h = entity2id[row["head"]]
        r = relation2id[row["relation"]]
        t = entity2id[row["tail"]]
        lines.append(f"{h}\t{r}\t{t}")

    out_path = PROCESSED_DIR / f"{split_name}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{split_name}.txt 边数: {len(lines)}")


def _write_queries(
    target_df: pd.DataFrame,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    split_name: str,
) -> None:
    """为 valid/test 构造 query JSON（只包含正例答案）。"""
    # complaint_id + relation -> list of tails
    answers: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for _, row in target_df.iterrows():
        cid = row["complaint_id"]
        rel = row["relation"]
        tail = row["tail"]
        answers[(cid, rel)].append(tail)

    queries: List[Dict] = []
    for (cid, rel), tails in answers.items():
        # head 在 triples 里就是 complaint 节点
        head_node = f"complaint:{cid}" if not any(
            t.startswith("complaint:") for t in tails
        ) else None

        # 如果 head 节点名在原始 triples 中已经是 complaint:XXX，则直接使用
        # 否则退回到 complaint:complaint_id 的约定
        if head_node not in entity2id:
            # 可能 triples 中本来就用了 complaint:XXX 形式
            # 在这种情况下，取该 complaint 任意一条目标边的 head 作为 head_node
            sample = target_df[target_df["complaint_id"] == cid].iloc[0]
            head_node = sample["head"]

        head_id = entity2id[head_node]
        rel_id = relation2id[rel]
        tail_ids = sorted({entity2id[t] for t in tails})
        answer_texts = sorted(set(tails))

        queries.append(
            {
                "complaint_id": cid,
                "head_id": head_id,
                "relation_id": rel_id,
                "answers": tail_ids,
                "head": head_node,
                "relation": rel,
                "answer_texts": answer_texts,
            }
        )

    out_path = PROCESSED_DIR / f"{split_name}_queries.json"
    out_path.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{split_name}_queries.json 条数: {len(queries)}，"
        f"诉求数: {len({q['complaint_id'] for q in queries})}"
    )


def main() -> None:
    df = load_and_clean_triples()

    entity2id, relation2id = build_mappings(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "entity2id.json").write_text(
        json.dumps(entity2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROCESSED_DIR / "relation2id.json").write_text(
        json.dumps(relation2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 entity2id.json / relation2id.json 到 {PROCESSED_DIR}")

    split_ids = split_complaints(df)
    write_complaint_split(split_ids)

    triples_to_graph_and_queries(df, split_ids, entity2id, relation2id)
    print("数据预处理与数据集构造完成。")


if __name__ == "__main__":
    main()

