import re
from pathlib import Path

import pandas as pd


# 相对项目根目录的路径推断：当前文件在 data/data_preprocess 下
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "combinnation.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_combinnation.xlsx"

# 原始 Excel 列名
ID_COL = "编号"
ENTITY_COL = "实体"
HIDDEN_COL = "隐患"
RISK_COL = "风险"
EVENT_COL = "事件"
OUTCOME_COL = "后果"

# 多种可能分隔符：中文/英文逗号、分号、斜杠、换行
SPLIT_PATTERN = re.compile(r"[，,;/\n\r]+")


def normalize_cell(x):
    """把单元格内容转为“取第一个值”的规范字符串；空则返回 None。"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    parts = [p.strip() for p in SPLIT_PATTERN.split(s) if p.strip()]
    if not parts:
        return None
    return parts[0]


def main() -> None:
    print(f"读取原始 Excel: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH, dtype=str)

    cols = [ID_COL, ENTITY_COL, HIDDEN_COL, RISK_COL, EVENT_COL, OUTCOME_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少以下列: {missing}，请检查 Excel 表头是否为: {cols}"
        )

    df = df[cols].copy()

    # 规范化所有列
    for col in cols:
        df[col] = df[col].apply(normalize_cell)

    # 编号为空 -> 丢行（必须有 complaint_id）
    before = len(df)
    df = df[~df[ID_COL].isna()]
    print(f"丢弃编号为空的行: {before - len(df)} 行")

    # 风险为空 -> 丢行（没有风险就无法参与风险预测）
    before = len(df)
    df = df[~df[RISK_COL].isna()]
    print(f"丢弃风险为空的行: {before - len(df)} 行")

    # 风险 == '空白' -> 整行丢弃（不作为训练数据）
    before = len(df)
    df = df[df[RISK_COL] != "空白"]
    print(f"丢弃 风险='空白' 的行: {before - len(df)} 行")

    # 事件为空 -> 填成 '空白'（保留行，只用一个占位节点）
    empty_event = df[EVENT_COL].isna().sum()
    df[EVENT_COL] = df[EVENT_COL].fillna("空白")
    print(f"事件为空并填充为 '空白' 的行数: {empty_event}")

    # 去重
    before = len(df)
    df = df.drop_duplicates()
    print(f"删除重复行: {before - len(df)} 行")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"清洗后数据已写入: {OUTPUT_PATH}")
    print(f"最终行数: {len(df)}")


if __name__ == "__main__":
    main()

