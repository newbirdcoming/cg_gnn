"""
将当前 processed 数据导出为多 sheet Excel：
  - train_triples / valid_triples / test_triples
  - valid_queries / test_queries

使用场景：
  - 便于第四章在论文/附录中展示训练/验证/测试数据内容
  - 快速检查切分、映射是否正确
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def _load_triples_txt(path: Path) -> pd.DataFrame:
    """
    输入格式：h\\tr\\tt
    输出：DataFrame 包含 head_id, relation_id, tail_id
    """
    rows: List[Tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        h, r, t = line.split("\t")
        rows.append((int(h), int(r), int(t)))

    df = pd.DataFrame(rows, columns=["head_id", "relation_id", "tail_id"])
    return df


def _add_text_columns(
    df: pd.DataFrame,
    entity_id2text: Dict[int, str],
    rel_id2text: Dict[int, str],
) -> pd.DataFrame:
    df = df.copy()
    df["head_text"] = df["head_id"].map(entity_id2text)
    df["relation_text"] = df["relation_id"].map(rel_id2text)
    df["tail_text"] = df["tail_id"].map(entity_id2text)
    return df


def _load_queries_json(path: Path) -> pd.DataFrame:
    """
    queries.json 的结构（来自 baseline.build_queries）：
      [
        {
          "complaint_id": "...",
          "head_id": 123,
          "relation_id": 4,
          "answers": [t1, t2, ...],
          "head": "complaint:xxx",
          "relation": "包含风险",
          "answer_texts": ["risk:..."]
        },
        ...
      ]
    """
    queries = _load_json(path)
    rows: List[Dict[str, Any]] = []

    for q in queries:
        answers = q.get("answers", [])
        answer_texts = q.get("answer_texts", []) or q.get("answer_texts", [])
        rows.append(
            {
                "complaint_id": q.get("complaint_id"),
                "head_id": q.get("head_id"),
                "relation_id": q.get("relation_id"),
                "head": q.get("head"),
                "relation": q.get("relation"),
                "answers": json.dumps(answers, ensure_ascii=False),
                "answers_texts": json.dumps(answer_texts, ensure_ascii=False),
                "answers_count": len(answers),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    out_path = PROCESSED_DIR / "splits_export.xlsx"
    if out_path.exists():
        # 覆盖旧文件（保证反复导出时不乱）
        out_path.unlink()

    entity2id = _load_json(PROCESSED_DIR / "entity2id.json")
    relation2id = _load_json(PROCESSED_DIR / "relation2id.json")
    entity_id2text = _invert_mapping(entity2id)
    rel_id2text = _invert_mapping(relation2id)

    # triples
    train_df = _load_triples_txt(PROCESSED_DIR / "train.txt")
    valid_df = _load_triples_txt(PROCESSED_DIR / "valid.txt")
    test_df = _load_triples_txt(PROCESSED_DIR / "test.txt")

    train_df = _add_text_columns(train_df, entity_id2text, rel_id2text)
    valid_df = _add_text_columns(valid_df, entity_id2text, rel_id2text)
    test_df = _add_text_columns(test_df, entity_id2text, rel_id2text)

    # queries
    valid_queries_df = _load_queries_json(PROCESSED_DIR / "valid_queries.json")
    test_queries_df = _load_queries_json(PROCESSED_DIR / "test_queries.json")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        train_df.to_excel(writer, sheet_name="train_triples", index=False)
        valid_df.to_excel(writer, sheet_name="valid_triples", index=False)
        test_df.to_excel(writer, sheet_name="test_triples", index=False)
        valid_queries_df.to_excel(writer, sheet_name="valid_queries", index=False)
        test_queries_df.to_excel(writer, sheet_name="test_queries", index=False)

    print(f"[export_splits_to_excel] Exported: {out_path}")


if __name__ == "__main__":
    main()

